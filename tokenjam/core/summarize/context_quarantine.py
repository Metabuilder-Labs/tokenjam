"""Reversible removal store for the context-audit page.

The page's Remove button never unlinks anything. It MOVES the file into a
quarantine directory and records where it came from, so the removal is undone
by moving it back. Hook entries (which are JSON inside a settings file, not
files of their own) are handled the same way: the whole settings file's prior
text is stashed here before the entry is stripped, and restoring rewrites it.

Two properties are load-bearing and deliberately different from
``summarize/backup.py``, which this module otherwise mirrors:

**The store outlives tj.** ``backup.py`` lives under ``summary_root(config)``
— i.e. beside the storage DB, inside ``~/.tj``. Anything stored there dies with
an uninstall that removes tj's state dir, which is exactly the wrong property
for a directory holding the user's only copy of a file they removed. So this
store is rooted at :data:`QUARANTINE_DIRNAME` under ``$HOME``, owned by nobody
but the user, named for its CONTENTS rather than for tj, and never placed under
a ``.tj*`` path an uninstaller might glob.

**A restore needs no tj at all.** Every record carries the original absolute
path and the quarantined file's name in plain JSON, and the payload is stored
uncompressed under its own name. A user with no working tj install can read
``manifest.json`` and move the files back by hand. That is the whole reason the
payload is not gzip'd the way ``backup.py`` gzips its originals: a store whose
recovery path requires the tool that is gone is not a recovery path.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenjam.core.summarize.session import SummarizeRefused, sha256
from tokenjam.utils.time_parse import utcnow

#: Home-relative name of the quarantine root. Named for what it holds, not for
#: tj — see the module docstring on surviving an uninstall.
QUARANTINE_DIRNAME = ".claude-context-trash"

#: Kinds of removal a record can describe.
KIND_FILE = "file"
KIND_HOOK = "hook"

_MANIFEST_NAME = "manifest.json"
_PAYLOAD_DIRNAME = "files"

_README = """\
This directory holds Claude Code context files removed from the tokenjam
context-audit page. Nothing here has been deleted.

manifest.json lists every removal: `original_path` is where the file came
from, `payload` is its name under files/. To restore something by hand,
copy files/<payload> back to its original_path. You do not need tokenjam
installed to do that.

Records with "kind": "hook" are different: the removal edited a settings
JSON file rather than removing a file. For those, files/<payload> is the
ENTIRE original settings file as it read before the edit, and restoring
means copying it back over original_path.
"""


def quarantine_root(home: Path | None = None) -> Path:
    return (home or Path.home()) / QUARANTINE_DIRNAME


def _payload_dir(root: Path) -> Path:
    return root / _PAYLOAD_DIRNAME


def _manifest_path(root: Path) -> Path:
    return root / _MANIFEST_NAME


def read_manifest(root: Path) -> list[dict[str, Any]]:
    """Every removal record, oldest first.

    A missing, unreadable, or non-list manifest reads as "no records" rather
    than raising: this feeds a GET the page polls, and a hand-mangled manifest
    must never 500 the page that is the only UI for fixing it.
    """
    f = _manifest_path(root)
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return [r for r in data] if isinstance(data, list) else []


def _write_manifest(root: Path, records: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _payload_dir(root).mkdir(parents=True, exist_ok=True)
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text(_README, encoding="utf-8")
    _manifest_path(root).write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _payload_name(record_id: str, original: Path) -> str:
    """A payload filename that is unique per record but still recognisable by a
    human reading ``files/`` without the manifest open."""
    return f"{record_id[:12]}-{original.name}"


@dataclass(frozen=True)
class Removal:
    """One reversible removal, as stored in the manifest."""

    record_id: str
    kind: str
    original_path: str
    payload: str
    removed_at: str
    label: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "kind": self.kind,
            "original_path": self.original_path, "payload": self.payload,
            "removed_at": self.removed_at, "label": self.label, "detail": self.detail,
        }


def _guard_removable(path: Path) -> None:
    """Refuse anything whose removal we cannot honestly reverse.

    ``is_symlink`` is checked before ``exists`` because ``exists`` follows the
    link: a broken symlink would otherwise be reported as a missing file, and a
    live one would have us move the TARGET while recording the link's path.
    """
    if path.is_symlink():
        raise SummarizeRefused(f"{path} is a symlink — refusing to remove through it.")
    if not path.exists():
        raise SummarizeRefused(f"{path} no longer exists — nothing to remove.")
    if path.is_dir():
        raise SummarizeRefused(f"{path} is a directory — this removes files, never trees.")


def remove_file(original: Path, *, label: str = "", detail: str = "",
                home: Path | None = None) -> Removal:
    """Move ``original`` into quarantine and record it. The file is never
    unlinked — a removal that cannot be undone is not what this page offers."""
    original = original.expanduser()
    _guard_removable(original)
    root = quarantine_root(home)
    record_id = sha256(f"{original}|{utcnow().isoformat()}")[:32]
    payload = _payload_name(record_id, original)
    _payload_dir(root).mkdir(parents=True, exist_ok=True)
    shutil.move(str(original), str(_payload_dir(root) / payload))
    rec = Removal(record_id, KIND_FILE, str(original), payload,
                  utcnow().isoformat(), label or original.name, detail)
    _write_manifest(root, [*read_manifest(root), rec.to_dict()])
    return rec


def remove_hook(settings_path: Path, event: str, matcher: str, command: str, *,
                label: str = "", home: Path | None = None) -> Removal:
    """Strip one hook entry from ``settings_path``, stashing the file's ENTIRE
    prior text first so a restore is a single copy-back.

    Refuses when the entry is not found rather than rewriting the file to an
    identical copy — a Remove that silently changes nothing reads as success
    and leaves the user believing a hook is gone while it still fires.
    """
    settings_path = settings_path.expanduser()
    _guard_removable(settings_path)
    try:
        before = settings_path.read_text(encoding="utf-8")
        data = json.loads(before)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummarizeRefused(f"{settings_path} is not readable JSON — refusing to edit it.") from exc
    if not isinstance(data, dict):
        raise SummarizeRefused(f"{settings_path} is not a JSON object — refusing to edit it.")

    events = data.get("hooks")
    removed = False
    if isinstance(events, dict) and isinstance(events.get(event), list):
        kept_matchers = []
        for entry in events[event]:
            if not isinstance(entry, dict) or entry.get("matcher", "") != matcher:
                kept_matchers.append(entry)
                continue
            hooks = entry.get("hooks")
            if not isinstance(hooks, list):
                kept_matchers.append(entry)
                continue
            kept = [h for h in hooks
                    if not (isinstance(h, dict) and str(h.get("command") or "") == command)]
            if len(kept) != len(hooks):
                removed = True
            # An entry whose last hook just went is dropped whole rather than
            # left as an empty matcher block the harness would still walk.
            if kept:
                kept_matchers.append({**entry, "hooks": kept})
        if removed:
            if kept_matchers:
                events[event] = kept_matchers
            else:
                events.pop(event, None)
    if not removed:
        raise SummarizeRefused(
            f"no {event} hook running that command is present in {settings_path} — "
            "nothing was changed (it may already have been removed).")

    root = quarantine_root(home)
    record_id = sha256(f"{settings_path}|{event}|{matcher}|{command}|{utcnow().isoformat()}")[:32]
    payload = _payload_name(record_id, settings_path)
    _payload_dir(root).mkdir(parents=True, exist_ok=True)
    (_payload_dir(root) / payload).write_text(before, encoding="utf-8")
    settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    rec = Removal(record_id, KIND_HOOK, str(settings_path), payload,
                  utcnow().isoformat(), label or f"{event} hook",
                  f"{event} hook removed from {settings_path.name}")
    _write_manifest(root, [*read_manifest(root), rec.to_dict()])
    return rec


def _restorable(root: Path, rec: dict[str, Any]) -> tuple[bool, str]:
    payload = _payload_dir(root) / str(rec.get("payload") or "")
    if not payload.exists():
        return False, "the quarantined copy is missing"
    target = Path(str(rec.get("original_path") or "")).expanduser()
    if not str(target):
        return False, "the record has no original path"
    if rec.get("kind") == KIND_HOOK:
        # A hook restore overwrites the settings file, which by definition
        # exists and has changed. It is restorable as long as we can write it.
        return (True, "") if target.parent.exists() else (False, "the settings file's directory is gone")
    if target.is_symlink():
        return False, "a symlink now sits at the original path"
    if target.exists():
        return False, "something already exists at the original path"
    return True, ""


def list_removals(home: Path | None = None) -> list[dict[str, Any]]:
    """Every removal, newest first, each with a computed ``restorable`` flag and
    the reason when it is false — the same conditions :func:`restore` enforces,
    surfaced up front so the page never offers a Restore that cannot run."""
    root = quarantine_root(home)
    out: list[dict[str, Any]] = []
    for rec in read_manifest(root):
        restorable, reason = _restorable(root, rec)
        out.append({**rec, "restorable": restorable, "reason": reason})
    out.sort(key=lambda r: str(r.get("removed_at") or ""), reverse=True)
    return out


def restore(record_id: str, home: Path | None = None) -> dict[str, Any]:
    """Put one removal back and drop its record."""
    root = quarantine_root(home)
    records = read_manifest(root)
    match = next((r for r in records if r.get("record_id") == record_id), None)
    if match is None:
        raise SummarizeRefused(f"no removal recorded under {record_id} — nothing to restore.")
    restorable, reason = _restorable(root, match)
    if not restorable:
        raise SummarizeRefused(f"cannot restore {match.get('original_path')}: {reason}.")

    payload = _payload_dir(root) / str(match["payload"])
    target = Path(str(match["original_path"])).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if match.get("kind") == KIND_HOOK:
        # The settings file still exists (we only edited it), so copy the prior
        # text back over it and drop the payload.
        shutil.copyfile(str(payload), str(target))
        payload.unlink(missing_ok=True)
    else:
        shutil.move(str(payload), str(target))
    _write_manifest(root, [r for r in records if r.get("record_id") != record_id])
    return {"record_id": record_id, "restored_path": str(target)}
