"""On-device quarantine for content a prune or an expire removes.

Prune and expire were the two largest unrealised recoveries in this tool and
neither had a write path, for a defensible reason: compression is reversible in
principle (the words are still there, shorter), while deletion is not. So the
tool only ever advised.

The answer is not to refuse to act. It is to make deletion RECOVERABLE, which is
what this module does: every fragment a prune or expire removes is written here,
verbatim, BEFORE the source file is touched. Removing content without a
committed quarantine entry is structurally impossible — the entry is written and
fsync'd first, its presence is re-read from disk, and the apply aborts untouched
if that read fails.

**What is recorded, and why each field is load-bearing.**

* ``source_path`` — where it came from, absolute.
* ``removed_text`` — the exact bytes, so a restore is an insertion of the
  original and not a re-rendering of it.
* ``start_line`` / ``end_line`` — where it sat at removal time. Advisory on
  restore, never trusted blindly: see below.
* ``source_sha256`` — the WHOLE source file as it was at removal time. This is
  the field that decides whether the line numbers still mean anything.
* ``route`` — prune or expire, so a user reading the list knows which decision
  produced the removal.
* ``removed_at`` — when.

**Why the hash matters more than the line numbers.** A restore that trusted a
stale line range would splice text into the middle of an unrelated rule, and a
corrupted instruction file is a strictly worse outcome than a failed restore: it
is silent, it ships, and the agent follows it. So a restore against a file whose
hash no longer matches does not guess. It re-anchors only when the exact
surrounding text can still be located unambiguously, and otherwise refuses with
the entry id and the path, so the user can paste it back themselves.

Lives under tj's own storage root beside the summarize backups
(``<storage parent>/summary/quarantine``), derived through
``session.summary_root`` rather than through a second config knob — one
derivation, so a user who moves ``storage.path`` moves all of it.
"""
from __future__ import annotations

import gzip
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize.session import SummarizeRefused, sha256, summary_root
from tokenjam.utils.time_parse import utcnow

#: How much surrounding text is stored to re-anchor a restore against a file
#: that changed. Enough to be unique in a normal instruction file; short enough
#: that an edit elsewhere in the same region does not destroy it.
ANCHOR_CHARS = 240


def quarantine_dir(config: TjConfig) -> Path:
    """``<tj storage root>/summary/quarantine``.

    Same derivation as ``backup.backups_dir`` on purpose: a user who repoints
    ``storage.path`` moves their backups and their quarantine together, and
    neither needs its own config knob.
    """
    return summary_root(config) / "quarantine"


@dataclass(frozen=True)
class QuarantineEntry:
    """One removed fragment, and everything needed to put it back."""

    entry_id: str
    source_path: str
    #: Line range the fragment occupied at removal time, 1-indexed, ``end``
    #: exclusive. Advisory: :func:`restore` re-derives the insertion point.
    start_line: int
    end_line: int
    #: sha256 of the WHOLE source file as it was at removal time. The single
    #: signal that decides whether the line range still means anything.
    source_sha256: str
    #: sha256 of the source file immediately AFTER the removal, so a restore can
    #: tell "unchanged since we cut it" from "edited since".
    result_sha256: str
    route: str
    removed_at: str
    removed_chars: int
    #: Verbatim text either side of the cut, for re-anchoring. Empty at a file
    #: boundary, which is a legitimate anchor of its own.
    before_anchor: str = ""
    after_anchor: str = ""
    #: Why this fragment was selected, in the words the user approved.
    reason: str = ""
    #: Populated on read; never written to the sidecar.
    removed_text: str = field(default="", compare=False)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("removed_text", None)
        return out


def source_after_cut(source_before: str, source_after: str, removed_text: str) -> int:
    """Where the removal left a hole in ``source_after``.

    Correct only for a SINGLE removal, which is the one case a caller cannot do
    better than: the common prefix of the two texts ends exactly at the cut.
    """
    limit = min(len(source_before), len(source_after))
    i = 0
    while i < limit and source_before[i] == source_after[i]:
        i += 1
    return i


def _blob_path(config: TjConfig, entry_id: str) -> Path:
    return quarantine_dir(config) / f"{entry_id}.text.gz"


def _meta_path(config: TjConfig, entry_id: str) -> Path:
    return quarantine_dir(config) / f"{entry_id}.meta.json"


def _fsync_write(path: Path, data: bytes) -> None:
    """Write ``data`` and force it to the platter before returning.

    A plain ``write_bytes`` returns once the bytes are in the page cache, which
    is fine for a backup and NOT fine here: the whole ordering guarantee is that
    the quarantine entry survives a crash that happens between the entry write
    and the source rewrite. Without the fsync the two writes can land in either
    order after a power loss, and the losing order is the one that deletes
    content with no record of it.
    """
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def record(
    config: TjConfig,
    *,
    source_path: str,
    removed_text: str,
    start_line: int,
    end_line: int,
    source_before: str,
    source_after: str,
    route: str,
    reason: str = "",
    before_anchor: str | None = None,
    after_anchor: str | None = None,
) -> QuarantineEntry:
    """Commit one removal to the quarantine and return its entry.

    Raises ``SummarizeRefused`` if the entry cannot be written or cannot be read
    back afterwards. The caller MUST treat that as fatal to the apply: an
    unverifiable quarantine entry is the same situation as no quarantine at all,
    and the point of this module is that the second situation cannot arise.
    """
    if not removed_text:
        raise SummarizeRefused("nothing to quarantine: the removal is empty.")
    entry_id = uuid.uuid4().hex[:16]
    # ANCHORS ARE MEASURED AGAINST THE RESULT, not against the original, and
    # this is not a detail. When one apply removes several fragments, a
    # fragment's neighbour in the ORIGINAL may itself be one of the removals —
    # so an anchor taken from the original names text that will not exist in the
    # file the restore has to find it in, and every multi-fragment restore
    # refuses. The caller computes the cut point in the RESULT and passes both
    # sides; the fallback below is only correct for a single removal, which is
    # exactly when the caller has nothing better to offer.
    if before_anchor is None or after_anchor is None:
        cut_at = source_after_cut(source_before, source_after, removed_text)
        before_anchor = source_after[max(0, cut_at - ANCHOR_CHARS):cut_at]
        after_anchor = source_after[cut_at:cut_at + ANCHOR_CHARS]
    entry = QuarantineEntry(
        entry_id=entry_id,
        source_path=str(Path(source_path).expanduser()),
        start_line=int(start_line),
        end_line=int(end_line),
        source_sha256=sha256(source_before),
        result_sha256=sha256(source_after),
        route=route,
        removed_at=utcnow().isoformat(),
        removed_chars=len(removed_text),
        before_anchor=before_anchor,
        after_anchor=after_anchor,
        reason=reason,
        removed_text=removed_text,
    )
    directory = quarantine_dir(config)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _fsync_write(_blob_path(config, entry_id), gzip.compress(removed_text.encode("utf-8")))
        _fsync_write(
            _meta_path(config, entry_id),
            json.dumps(entry.to_dict(), ensure_ascii=False, indent=1).encode("utf-8"),
        )
    except OSError as exc:
        raise SummarizeRefused(
            f"could not write the quarantine entry ({exc}). Nothing was removed — "
            f"content is never cut without a committed way back."
        ) from exc

    # Read it back FROM DISK. A successful write call is not evidence the entry
    # is retrievable, and "we wrote it" is exactly the belief that would let an
    # unrecoverable deletion through.
    verified = read(config, entry_id)
    if verified is None or verified.removed_text != removed_text:
        raise SummarizeRefused(
            f"the quarantine entry for {source_path} could not be read back after "
            f"writing it. Nothing was removed."
        )
    return entry


def read(config: TjConfig, entry_id: str) -> QuarantineEntry | None:
    """One entry with its text, or ``None`` when it is missing or unreadable.

    Tolerant on purpose: a corrupt sidecar reads as absent so a listing can
    never be crashed by one bad file — the same discipline
    ``backup.recorded_output`` applies.
    """
    meta_file = _meta_path(config, entry_id)
    blob_file = _blob_path(config, entry_id)
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        text = gzip.decompress(blob_file.read_bytes()).decode("utf-8")
    except (OSError, ValueError, gzip.BadGzipFile):
        return None
    if not isinstance(meta, dict):
        return None
    fields = {k: meta.get(k) for k in QuarantineEntry.__dataclass_fields__ if k != "removed_text"}
    try:
        return QuarantineEntry(**fields, removed_text=text)  # type: ignore[arg-type]
    except TypeError:
        return None


def list_entries(config: TjConfig, *, source_path: str | None = None) -> list[QuarantineEntry]:
    """Every readable entry, newest first, optionally scoped to one file."""
    directory = quarantine_dir(config)
    if not directory.is_dir():
        return []
    out: list[QuarantineEntry] = []
    for meta_file in sorted(directory.glob("*.meta.json")):
        entry = read(config, meta_file.name[: -len(".meta.json")])
        if entry is None:
            continue
        if source_path and entry.source_path != str(Path(source_path).expanduser()):
            continue
        out.append(entry)
    out.sort(key=lambda e: e.removed_at, reverse=True)
    return out


def _anchor_point(current: str, entry: QuarantineEntry) -> tuple[int, str] | None:
    """Where the fragment goes in ``current``, and how confidently, or ``None``.

    Three anchors, tried strongest first. Each must be UNAMBIGUOUS — exactly one
    occurrence — because an anchor matching twice is an anchor that does not
    know which of two places it means, and picking one is guessing.

    1. **Both sides.** The text that sat immediately before and immediately after
       the cut, adjacent. This is the strongest signal available and is what
       resolves when nothing around the hole has been edited.
    2. **The text before it.** Survives any edit made AFTER the cut point, which
       includes another fragment from the same apply being restored first. Only
       this one makes a multi-fragment restore work.
    3. **The text after it.** The mirror, for a fragment cut from the very top of
       a file, which has no preceding text to anchor to.
    """
    for anchor, offset, label in (
        (entry.before_anchor + entry.after_anchor, len(entry.before_anchor), "both sides"),
        (entry.before_anchor, len(entry.before_anchor), "the text before it"),
        (entry.after_anchor, 0, "the text after it"),
    ):
        if anchor and current.count(anchor) == 1:
            return current.find(anchor) + offset, label
    return None


def _reinsert(current: str, entry: QuarantineEntry) -> tuple[str | None, str]:
    """``(restored text, "")`` or ``(None, reason)``.

    A stale LINE NUMBER is never used. Inserting at one would splice the
    fragment into the middle of whatever now occupies those lines, and a
    corrupted instruction file is the worse failure by a distance: it is silent,
    it ships, and the agent follows it. So the fragment is placed by matching
    the text that surrounded it, and when that cannot be done unambiguously this
    refuses and says what to do instead.

    The recorded hash is what distinguishes "unchanged since we cut it" (the
    restore is exact) from "edited since" (the restore is a re-anchor). Both can
    succeed; only the first is byte-for-byte the original file.
    """
    if entry.removed_text in current:
        return None, "that text is already present in the file — nothing to restore."

    placed = _anchor_point(current, entry)
    if placed is None:
        if sha256(current) == entry.result_sha256:
            # Unchanged since the removal but nothing to anchor to: the fragment
            # was the whole file's content. The recorded line range IS still
            # exact here, which the hash is what proves.
            lines = current.splitlines(keepends=True)
            at_line = max(0, min(entry.start_line - 1, len(lines)))
            head = "".join(lines[:at_line])
            return head + entry.removed_text + "".join(lines[at_line:]), ""
        return None, (
            f"{entry.source_path} has changed since this fragment was removed and "
            f"the text that used to surround it can no longer be located "
            f"unambiguously. Refusing to insert at a stale line number rather "
            f"than risk splicing it into the wrong place. The text is intact — "
            f"read it with `tj summarize quarantine show {entry.entry_id}` and "
            f"paste it where it belongs."
        )
    at, _label = placed
    return current[:at] + entry.removed_text + current[at:], ""


def restore(config: TjConfig, entry_id: str, *, go: bool = False) -> dict:
    """Put one quarantined fragment back. Default dry-run; ``go`` writes.

    Returns ``{"entry_id", "restored", "dry_run", "reason", "preview"}``.
    Refuses rather than corrupting — see :func:`_reinsert`.
    """
    from tokenjam.core.summarize.apply import _owned_by_current_user, _write

    entry = read(config, entry_id)
    if entry is None:
        raise SummarizeRefused(
            f"no quarantine entry {entry_id!r}. `tj summarize quarantine list` "
            f"shows what is there."
        )
    path = Path(entry.source_path)
    if path.is_symlink():
        raise SummarizeRefused(
            f"{path} is a symlink — refusing to write through it."
        )
    if not path.is_file():
        raise SummarizeRefused(
            f"{path} no longer exists, so there is nothing to restore into. The "
            f"text is still in the quarantine."
        )
    if not _owned_by_current_user(path):
        raise SummarizeRefused(f"{path} is owned by another user — refusing to write.")

    current = path.read_text(encoding="utf-8")
    unchanged = sha256(current) == entry.result_sha256
    restored, reason = _reinsert(current, entry)
    if restored is None:
        return {
            "entry_id": entry_id, "restored": False, "dry_run": not go,
            "reason": reason, "path": str(path), "preview": "",
            "source_unchanged": unchanged,
        }
    if go:
        # A restore is a WRITE, and it was the only write rail in this feature
        # without a backup. `_anchor_point` can legitimately find a unique match
        # in a heavily-edited file that is not where the fragment belongs any
        # more; that insertion is correct by the anchor's rules and still wrong
        # by the reader's, and without this it could not be reversed with
        # `tj summarize undo` the way every other write here can.
        from tokenjam.core.summarize import backup

        backup.save(config, str(path), original=current, output=restored)
        _write(path, restored)
    return {
        "entry_id": entry_id, "restored": go, "dry_run": not go, "reason": "",
        "path": str(path), "preview": entry.removed_text,
        "source_unchanged": unchanged,
    }


def restore_all(config: TjConfig, *, source_path: str | None = None, go: bool = False) -> dict:
    """Restore every entry (optionally for one file), LAST CUT FIRST.

    The order is load-bearing, not cosmetic. Two fragments removed from one file
    are each anchored on the text that surrounded them in the RESULT, and
    restoring the later one first leaves every earlier one's preceding text
    exactly as the anchor recorded it. The reverse order inserts text into the
    middle of the next fragment's anchor and makes it unresolvable.
    """
    results = []
    entries = list_entries(config, source_path=source_path)
    entries.sort(key=lambda e: (e.source_path, e.removed_at, e.start_line), reverse=True)
    for entry in entries:
        results.append(restore(config, entry.entry_id, go=go))
    return {
        "restored": sum(1 for r in results if r["restored"]),
        "refused": [r for r in results if not r["restored"] and r["reason"]],
        "dry_run": not go,
        "results": results,
    }


def forget(config: TjConfig, entry_id: str) -> bool:
    """Drop one entry. Returns whether anything was removed.

    Deliberately NOT called by restore: a restored fragment's entry stays, so a
    user who restores and then decides they were right the first time still has
    the record of what was cut.
    """
    removed = False
    for path in (_meta_path(config, entry_id), _blob_path(config, entry_id)):
        try:
            path.unlink()
            removed = True
        except OSError:
            pass
    return removed


__all__ = [
    "ANCHOR_CHARS",
    "QuarantineEntry",
    "forget",
    "list_entries",
    "quarantine_dir",
    "read",
    "record",
    "restore",
    "restore_all",
]
