"""Apply staged summarize results to their files (DEC-025/026) + undo.

``apply_staged`` is take-all over the staging dir: for each staged result it
owner-checks, hash-guards (skipping files changed since ``check``), backs up the
original, and atomically writes the restored text (preserving the file's mode).
The **default is a dry-run**; ``go=True`` writes. ``undo`` restores a backup,
refusing on post-apply drift. No flag bypasses a guard (DEC-027 agent-safety).
"""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize import backup
from tokenjam.core.summarize.session import SummarizeRefused, clear, list_staged, read_staged, sha256


def _owned_by_current_user(p: Path) -> bool:
    if not hasattr(os, "getuid"):          # non-POSIX (Windows) — no ownership model to honour
        return True
    return p.stat().st_uid == os.getuid()


def _atomic_write(p: Path, text: str) -> None:
    """Write ``text`` to ``p`` atomically (temp in the same dir → ``os.replace``), preserving mode."""
    mode = stat.S_IMODE(p.stat().st_mode)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tj-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def apply_staged(config: TjConfig, path: str | None = None, *, go: bool = False) -> dict:
    """Apply staged results (all, or just ``path``). Default dry-run; ``go`` writes.

    Returns ``{"applied": [...], "skipped": [{"path", "reason"}], "dry_run": bool}``.
    Each file is owner-checked + hash-guarded; drifted / unowned / not-structure-ok
    files are skipped and reported, never written.
    """
    if path is not None:
        one = read_staged(config, path)
        if one is None:
            return {"applied": [], "skipped": [{"path": path, "reason": "not staged"}],
                    "dry_run": not go}
        entries = [one]
    else:
        entries = list_staged(config)

    applied: list[dict] = []
    skipped: list[dict] = []
    for e in entries:
        sp = e["path"]
        p = Path(sp).expanduser()
        if not e["structure_ok"]:
            skipped.append({"path": sp, "reason": "structure check did not pass"})
            continue
        if not p.is_file():
            skipped.append({"path": sp, "reason": "file not found"})
            continue
        if p.is_symlink():
            skipped.append({"path": sp, "reason": "symlink — refusing to rewrite through it"})
            continue
        if not _owned_by_current_user(p):
            skipped.append({"path": sp, "reason": "owned by another user — refusing to rewrite"})
            continue
        current = p.read_text(encoding="utf-8")
        if sha256(current) != e["source_sha256"]:
            skipped.append({"path": sp, "reason": "changed since check — re-prep it"})
            continue
        if go:
            backup.save(config, sp, original=current, output=e["restored"],
                        est_tokens_saved=int(e.get("est_tokens_saved", 0) or 0))
            _atomic_write(p, e["restored"])
            clear(config, sp)
        applied.append({"path": sp, "est_tokens_saved": e["est_tokens_saved"], "diff": e["diff"]})
    return {"applied": applied, "skipped": skipped, "dry_run": not go}


def undo(config: TjConfig, path: str, *, go: bool = False) -> dict:
    """Restore the backup for ``path``. Default dry-run; ``go`` writes.

    Raises ``SummarizeRefused`` if there is no backup or the file changed since apply.
    """
    p = Path(path).expanduser()
    if p.is_symlink():
        raise SummarizeRefused(
            f"{path} is a symlink — undo won't restore through a link (it would replace the link, "
            f"not the file that was summarized). Point it at the real file.")
    current = p.read_text(encoding="utf-8") if p.is_file() else None
    original = backup.load_original(config, str(p), current)   # raises on drift / missing backup
    if go:
        if p.exists():
            _atomic_write(p, original)
        else:
            p.write_text(original, encoding="utf-8")           # file was deleted — recreate it
    return {"path": str(p), "restored": go, "dry_run": not go}
