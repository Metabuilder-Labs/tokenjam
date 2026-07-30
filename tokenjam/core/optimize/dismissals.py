"""Which cards the user has dismissed — server-side, durable, and reversible.

Dismissal used to live in the browser (`localStorage`), which means it was not
really recorded at all: a cleared profile, a second browser or a different
machine and every dismissed card came back. That is a decision the user made,
and a product that forgets it is asking them to make it again.

**Why a JSON sink beside the DB rather than a DuckDB table.** The applied and
reverted records already live this way, and the reason is structural rather
than historical: `tj rules` is in ``no_db_commands`` and must answer while
`tj serve` holds the DuckDB write lock, and DuckDB permits one writer OR many
readers across processes with no read-only escape hatch (see the CLI data-access
note in the repo docs). A dismissal stored in DuckDB could therefore not be
READ by the command that has to honour it whenever the daemon is up. The
founder's ask was that this stop living in the browser; this is the
server-side, durable home the sibling decisions already use, and putting it
anywhere else would be a third storage shape for one class of record.

**Dismissal suppresses the OFFER and nothing else** (Critical Rule 32). The
behaviour still happened and still cost what it cost; the user saying "not this
one" is a statement about our recommendation, not about their bill. Every
`past_overspend_*` figure is untouched, and a test asserts the record is
byte-identical with and without a dismissal.

**A dismissal must be reversible.** A durable dismissal that cannot be undone
is worse than a transient one — the user loses the card permanently with no way
back, and the durability is what makes that permanent. So a dismissal is a
record with a state, exactly like an apply, and :func:`undismiss` restores it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenjam.core.optimize.relearn_apply import _storage_base_dir
from tokenjam.utils.time_parse import utcnow

#: Same state vocabulary the apply ledgers use, so "is this record live" is one
#: question with one answer across all three.
STATE_DISMISSED = "dismissed"
STATE_RESTORED = "restored"


def dismissals_path(config: Any) -> Path:
    """``<storage-parent>/dismissed.json`` — honours ``--config`` /
    ``storage.path`` exactly like its apply-ledger siblings, so a throwaway
    ``--db`` never writes to the operator's real ``~/.tj``."""
    return _storage_base_dir(config) / "dismissed.json"


def _load(config: Any) -> list[dict]:
    path = dismissals_path(config)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Unreadable reads as "nothing dismissed", which leaves every card on
        # offer. That direction can only waste attention; the opposite would
        # hide cards the user never dismissed.
        return []
    return raw if isinstance(raw, list) else []


def _write(config: Any, records: list[dict]) -> None:
    path = dismissals_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_dismissals(config: Any) -> list[dict]:
    """Every dismissal record, live or restored."""
    return _load(config)


def dismissed_signatures(config: Any) -> set[str]:
    """The signatures currently dismissed.

    Same shape and same exclusion rule as ``cost_apply.applied_signatures`` and
    its relearn sibling — a restored record is excluded, so an un-dismiss puts
    the card back exactly as a revert re-opens an applied one. Stated here once
    rather than inline at each call site, for the reason the applied set is:
    three copies of a ledger filter is three chances for one to forget the
    exclusion and keep hiding a card the user brought back.
    """
    return {
        str(rec.get("signature") or "")
        for rec in _load(config)
        if rec.get("state") != STATE_RESTORED and rec.get("signature")
    }


def dismiss(config: Any, signature: str, *, reason: str = "") -> dict:
    """Record that the user dismissed ``signature``. Idempotent.

    ``reason`` is optional and free-text; it is stored, never interpreted. An
    existing live dismissal is returned unchanged rather than duplicated.
    """
    signature = str(signature or "").strip()
    if not signature:
        raise ValueError("a dismissal needs a signature.")
    records = _load(config)
    for rec in records:
        if rec.get("signature") == signature and rec.get("state") != STATE_RESTORED:
            return rec
    record = {
        "signature": signature,
        "state": STATE_DISMISSED,
        "reason": str(reason or ""),
        "dismissed_at": utcnow().isoformat(),
        "restored_at": None,
    }
    records.append(record)
    _write(config, records)
    return record


def undismiss(config: Any, signature: str) -> dict | None:
    """Bring a dismissed card back. Returns the restored record, or ``None``.

    The half that makes durable dismissal safe to offer at all: without it the
    user trades a card that came back on every browser for one that never
    comes back anywhere.
    """
    signature = str(signature or "").strip()
    records = _load(config)
    restored: dict | None = None
    for rec in records:
        if rec.get("signature") == signature and rec.get("state") != STATE_RESTORED:
            rec["state"] = STATE_RESTORED
            rec["restored_at"] = utcnow().isoformat()
            restored = rec
    if restored is not None:
        _write(config, records)
    return restored


__all__ = [
    "STATE_DISMISSED",
    "STATE_RESTORED",
    "dismiss",
    "dismissals_path",
    "dismissed_signatures",
    "list_dismissals",
    "undismiss",
]
