"""Append-only savings sink for the `tj hook cap-output` hook.

The hook must **never take a DuckDB lock** (it fires on every tool call, and the
`tj serve` daemon may hold the write lock — cf. #61). So trims are recorded to a
lightweight append-only JSONL sink next to the storage DB, and the full original
output is persisted to a sibling file so a trim is always *recoverable*. A
separate offline path (`tj savings`, `tj context`) reads the sink to show
cumulative reclaimed tokens.

Everything here is **fail-safe**: a write that fails is swallowed (the hook's
contract is fail-open — a tj bug must never lose data or break a session). The
path derives from `config.storage.path`'s parent (mirroring
`summarize.session.summary_root`) so `--db` / `TJ_CONFIG` overrides isolate it —
which the A/B harness relies on.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

from tokenjam.utils.time_parse import utcnow


def hooks_dir(config) -> Path:
    """`<storage-parent>/hooks/` — sink dir, next to the DB, honoring overrides."""
    sp = getattr(getattr(config, "storage", None), "path", "") or ""
    base = Path.home() / ".tj" if sp in ("", ":memory:") else Path(sp).expanduser().parent
    return base / "hooks"


def savings_path(config) -> Path:
    """The append-only JSONL of trim events."""
    return hooks_dir(config) / "cap_output.jsonl"


def preserved_dir(config) -> Path:
    """Where full pre-trim outputs are saved so trims stay recoverable."""
    return hooks_dir(config) / "outputs"


def _sanitize(tool: str) -> str:
    return "".join(c for c in (tool or "tool") if c.isalnum() or c in "-_")[:24] or "tool"


def persist_output(config, tool: str, session_id: str, text: str) -> Path | None:
    """Write the full pre-trim output to disk; return its path (or None on error).

    Filename is unique without a clock dependency in the hot path proper:
    ``<iso-compact>-<tool>-<pid>.txt``. Fail-safe — returns None if anything
    goes wrong so the caller falls back to a re-run hint in the marker.
    """
    try:
        d = preserved_dir(config)
        d.mkdir(parents=True, exist_ok=True)
        ts = utcnow().strftime("%Y%m%dT%H%M%S%f")
        sid = (session_id or "nosession")[:12]
        fname = f"{ts}-{_sanitize(tool)}-{sid}-{os.getpid()}.txt"
        p = d / fname
        p.write_text(text)
        return p
    except Exception:
        return None


def append_saving(config, event: dict) -> None:
    """Append one trim event to the JSONL sink. Fail-safe (never raises).

    The `ts` is stamped here (UTC, tz-aware — CLAUDE.md Rule 9). `default=str`
    keeps it robust to any non-serializable value (mirrors alerts.FileChannel).
    """
    try:
        p = savings_path(config)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = dict(event)
        record.setdefault("ts", utcnow().isoformat())
        with open(p, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def read_savings(config, session_id: str | None = None) -> list[dict]:
    """Read all trim events (optionally scoped to a session). Tolerant of a
    missing file and of partially-written lines (append-only can race a read)."""
    p = savings_path(config)
    out: list[dict] = []
    try:
        if not p.exists():
            return out
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if session_id and rec.get("session_id") != session_id:
                continue
            out.append(rec)
    except Exception:
        return out
    return out


def summarize_savings(events: list[dict]) -> dict:
    """Aggregate trim events into totals for the `tj savings` / `tj context`
    surfaces. All token counts are char/4 estimates — callers must label them
    "estimated"/"reclaimed" (honesty discipline)."""
    total_saved = 0
    total_orig = 0
    n = 0
    by_tool: dict[str, dict] = {}
    by_session: dict[str, int] = {}
    today = utcnow().date().isoformat()
    saved_today = 0
    for e in events:
        saved = int(e.get("saved_tok_est", 0) or 0)
        orig = int(e.get("orig_tok_est", 0) or 0)
        total_saved += saved
        total_orig += orig
        n += 1
        tool = str(e.get("tool", "?"))
        bt = by_tool.setdefault(tool, {"trims": 0, "saved_tok_est": 0})
        bt["trims"] += 1
        bt["saved_tok_est"] += saved
        sid = str(e.get("session_id", "?"))
        by_session[sid] = by_session.get(sid, 0) + saved
        ts = str(e.get("ts", ""))
        if ts[:10] == today:
            saved_today += saved
    return {
        "trims": n,
        "saved_tok_est": total_saved,
        "orig_tok_est": total_orig,
        "saved_today_tok_est": saved_today,
        "by_tool": by_tool,
        "by_session": by_session,
    }


def _as_dict(obj) -> dict:
    return asdict(obj) if is_dataclass(obj) else dict(obj)
