"""Mark a cost proposal "applied" — and nothing more.

A cost proposal is advise-only (see ``cost_proposals``): its fix lives in the
user's own application code, which tokenjam has no workspace to write into. So
unlike ``relearn_apply`` there is NO file write, no delivery routing, no git commit
here. "Apply" means exactly one thing: the user tells tokenjam "I made this
change".

That "I changed something at time T" marker is the loop primitive
``core.loop.Expectation`` already models. Marking a cost proposal applied
therefore:

  1. creates an ``Expectation`` whose ``created_at`` is the "applied at T"
     marker and whose ``agent_id`` scopes it, and
  2. appends a record to a small ``cost_applied.json`` ledger (mirroring
     ``relearn_apply``'s ``applied_fixes.json``) carrying the proposal
     snapshot + its ``target_key``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tokenjam.core.optimize.relearn_apply import _storage_base_dir


class CostApplyRefused(Exception):
    """Raised when a mark-applied request can't be honored (bad payload, no DB).
    The API layer translates this to a 409, mirroring RelearnApplyRefused."""


def cost_applied_path(config: Any) -> Path:
    """``<storage-parent>/cost_applied.json`` — honors --config / storage.path
    exactly like ``relearn_apply.applied_fixes_path`` (never the real ~/.tj for
    a :memory: / "" storage.path)."""
    return _storage_base_dir(config) / "cost_applied.json"


@dataclass
class CostAppliedRecord:
    id:             str                    # == the linked expectation_id
    expectation_id: str
    signature:      str
    analyzer:       str                    # downsize | cache | trim
    kind:           str                    # always "cost"
    title:          str
    target_key:     dict[str, Any]
    agent_id:       str
    applied_at:     str                    # the fix marker (Expectation.created_at)
    baseline:       dict[str, Any]
    past_overspend_usd:    float | None
    past_overspend_tokens: int | None
    estimate_basis: str
    state:          str = "applied"        # applied | reverted
    reverted_at:    str | None = None
    # Historical scaffold from the (removed) verify pass — left in place so an
    # existing ``cost_applied.json`` on disk still deserializes; nothing reads
    # past the empty defaults below anymore.
    verify: dict[str, Any] = field(default_factory=lambda: {
        "realized_usd_delta": None, "realized_tokens_delta": None,
        "pre_value": None, "post_value": None,
        "pre_sessions": None, "post_sessions": None,
        "verdict": None, "reason": None, "estimate_basis": None,
        "last_checked_at": None,
    })

    def to_dict(self) -> dict:
        return asdict(self)


def _load_ledger(config: Any) -> list[dict]:
    p = cost_applied_path(config)
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return raw if isinstance(raw, list) else []


def _write_ledger(config: Any, records: list[dict]) -> None:
    p = cost_applied_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def list_applied(config: Any) -> list[dict]:
    return _load_ledger(config)


def applied_signatures(config: Any) -> set[str]:
    """The signatures a user has already told tokenjam they dealt with.

    THE derivation of that set — "non-reverted records in ``cost_applied.json``"
    — stated once. It was written out inline at two call sites (the Review
    inbox route and the CLI's own rollup) and a third arrived with the
    rule-write surface; three copies of a ledger filter is three chances for
    one of them to keep counting a reverted record and silently re-hide a card
    the user re-opened. Pair it with :func:`signature_is_applied`, which owns
    the MATCHING half, so neither the set nor the matcher can drift alone.

    Reverted records are excluded on purpose: a revert is the user saying the
    fix is no longer in place, so the proposal is open again.
    """
    return {
        str(rec.get("signature") or "")
        for rec in list_applied(config)
        if rec.get("state") != "reverted" and rec.get("signature")
    }


def signature_is_applied(signature: str, applied_sigs: set[str]) -> bool:
    """Whether ``signature`` should read as already applied against
    ``applied_sigs`` (the signatures on non-reverted ``cost_applied.json``
    records).

    Exact match is the common case. One legacy fallback exists: the
    ``downsize`` per-agent proposal signature has been through TWO formats.
    It started as agent-only (``cost:downsize:<agent>``), then grew to
    ``cost:downsize:<agent>:<provider>:<model>:<alt_model>`` (see
    ``_downsize_agent_proposals`` in ``cost_proposals.py``) so that an agent
    running two over-sized models gets two distinct signatures instead of
    colliding in ``past_overspend_rollup``'s dedup-by-signature. That change
    deliberately did not migrate the stored ledger (its own commit message
    says so): "an existing apply/dismiss on one of them will not match after
    the upgrade." A mark recorded under the OLD agent-only signature must
    still count as "applied" for every model-qualified signature of that same
    agent -- there is no way to tell, after the fact, which of an agent's
    model rows the mark was originally for, and reading it as "still open"
    would wrongly re-nag the user for a fix they already told tokenjam about.
    This resolves that ambiguity toward "still applied" (the same ambiguity
    the old signature already had, just carried forward) rather than
    surprising the user by reopening a fixed card.
    """
    if signature in applied_sigs:
        return True
    parts = signature.split(":")
    if len(parts) >= 3 and parts[0] == "cost" and parts[1] == "downsize":
        legacy = ":".join(parts[:3])
        return legacy in applied_sigs
    return False


def get_applied(config: Any, record_id: str) -> dict | None:
    for rec in _load_ledger(config):
        if rec.get("id") == record_id:
            return rec
    return None


def mark_applied(
    db_or_conn: Any,
    config: Any,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """Record that the user applied a cost proposal.

    ``proposal`` is a STORED CostProposal dict, resolved server-side by its
    ``proposal_id`` (``relearn_proposals.list_cost_proposals``) — never a
    proposal echoed back by the caller, same guard the relearn apply path
    uses, so every value in the ledger record is one the detector produced.
    Creates the Expectation marker + appends the ledger record. Idempotent
    per signature: an existing non-reverted record for the same signature is
    returned unchanged rather than duplicated.

    Raises CostApplyRefused when the proposal is malformed or the DB is
    unavailable (the marker needs a real ``expectations`` table).
    """
    from tokenjam.core.loop import create_expectation

    signature = str(proposal.get("signature") or "").strip()
    if not signature:
        raise CostApplyRefused("proposal has no signature.")

    for rec in _load_ledger(config):
        if rec.get("state") == "reverted":
            continue
        rec_signature = str(rec.get("signature") or "")
        if signature_is_applied(signature, {rec_signature}):
            return rec  # already marked — don't double-mark or double-count

    agent_id = str(proposal.get("agent_id") or "").strip() or None
    title = str(proposal.get("title") or signature)[:200]
    advise = str(proposal.get("advise_text") or "")

    try:
        exp = create_expectation(
            db_or_conn,
            name=title,
            description=advise or None,
            agent_id=agent_id,
        )
    except Exception as exc:
        raise CostApplyRefused(f"could not create the fix marker: {exc}") from exc

    record = CostAppliedRecord(
        id=exp.expectation_id,
        expectation_id=exp.expectation_id,
        signature=signature,
        analyzer=str(proposal.get("analyzer") or ""),
        kind="cost",
        title=title,
        target_key=dict(proposal.get("target_key") or {}),
        agent_id=agent_id or "",
        applied_at=exp.created_at.isoformat() if exp.created_at else "",
        baseline=dict(proposal.get("baseline") or {}),
        past_overspend_usd=proposal.get("past_overspend_usd"),
        past_overspend_tokens=proposal.get("past_overspend_tokens"),
        estimate_basis=str(proposal.get("estimate_basis") or ""),
    )
    records = _load_ledger(config)
    records.append(record.to_dict())
    _write_ledger(config, records)
    return record.to_dict()


def revert_applied(config: Any, record_id: str) -> dict:
    """Mark a cost record reverted (the user undid their change). No file to
    restore — advise-only — so this just flips ``state`` so the ledger stops
    counting its realized delta. Raises CostApplyRefused on an unknown id."""
    from tokenjam.utils.time_parse import utcnow

    records = _load_ledger(config)
    for rec in records:
        if rec.get("id") == record_id:
            rec["state"] = "reverted"
            rec["reverted_at"] = utcnow().isoformat()
            _write_ledger(config, records)
            return rec
    raise CostApplyRefused(f"no cost_applied record {record_id}.")
