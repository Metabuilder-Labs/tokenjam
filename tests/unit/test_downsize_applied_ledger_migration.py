"""The downsize per-agent proposal signature went through two formats:
``cost:downsize:<agent>`` -> ``cost:downsize:<agent>:<provider>:<model>:
<alt_model>`` (``db4e1143`` -- so an agent running two over-sized models
gets two distinct signatures instead of colliding in
``past_overspend_rollup``'s dedup-by-signature).

That change deliberately did NOT migrate the stored ``cost_applied.json``
ledger -- its own commit message says so: "an existing apply/dismiss on one
of them will not match after the upgrade." Left as-is, a user who already
told tokenjam "I applied this fix" for an agent under the OLD signature
would see that same agent's card reopen as unfixed under the NEW signature,
which reads as tokenjam forgetting a fix it already knew about.

``cost_apply.signature_is_applied`` closes that gap: a legacy agent-only
mark still counts as "applied" for every model-qualified signature of that
same agent. There is no way to tell, after the fact, which of an agent's
model rows the old mark was for, so this resolves the ambiguity toward
"still applied" (never surprising the user by reopening a fixed card) --
the FORMAT itself is untouched here; only the migration is this file's
concern.
"""
from __future__ import annotations

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import cost_apply
from tokenjam.utils.time_parse import utcnow


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


# --- signature_is_applied: the pure predicate --------------------------------

def test_exact_match():
    assert cost_apply.signature_is_applied(
        "cost:downsize:svc-a:anthropic:claude-opus-4-8:claude-haiku-4-5",
        {"cost:downsize:svc-a:anthropic:claude-opus-4-8:claude-haiku-4-5"},
    )


def test_legacy_agent_only_mark_covers_every_model_qualified_signature():
    legacy = {"cost:downsize:svc-a"}
    assert cost_apply.signature_is_applied(
        "cost:downsize:svc-a:anthropic:claude-opus-4-8:claude-haiku-4-5", legacy,
    )
    assert cost_apply.signature_is_applied(
        "cost:downsize:svc-a:anthropic:claude-sonnet-4-5:claude-haiku-4-5", legacy,
    )


def test_never_matches_a_different_agent():
    legacy = {"cost:downsize:svc-a"}
    assert not cost_apply.signature_is_applied(
        "cost:downsize:svc-b:anthropic:claude-opus-4-8:claude-haiku-4-5", legacy,
    )


def test_never_matches_a_different_analyzer():
    # A non-downsize signature must never fall into the downsize-only shim.
    assert not cost_apply.signature_is_applied(
        "cost:subagent:svc-a:anthropic:claude-opus-4-8", {"cost:subagent:svc-a"},
    )


def test_no_false_positive_with_no_applied_records_at_all():
    assert not cost_apply.signature_is_applied(
        "cost:downsize:svc-a:anthropic:claude-opus-4-8:claude-haiku-4-5", set(),
    )


# --- mark_applied's own idempotency check honors the same migration ---------

def test_mark_applied_recognizes_a_pre_existing_legacy_record(cfg, db):
    """A ledger written before db4e1143 carries the agent-only signature.
    Calling mark_applied with the NEW model-qualified signature for that
    same agent must return the EXISTING record, not create a duplicate --
    otherwise a user who already applied the fix sees it reopened."""
    legacy_record = {
        "id": "exp-legacy", "expectation_id": "exp-legacy",
        "signature": "cost:downsize:svc-a",
        "analyzer": "downsize", "kind": "cost", "title": "legacy",
        "target_key": {}, "agent_id": "", "applied_at": utcnow().isoformat(),
        "baseline": {}, "past_overspend_usd": 3.0, "past_overspend_tokens": 1000,
        "estimate_basis": "", "state": "applied", "reverted_at": None,
        "verify": {},
    }
    cost_apply._write_ledger(cfg, [legacy_record])

    result = cost_apply.mark_applied(db, cfg, {
        "signature": "cost:downsize:svc-a:anthropic:claude-opus-4-8:claude-haiku-4-5",
        "analyzer": "downsize", "title": "svc-a", "target_key": {},
        "agent_id": "", "baseline": {},
        "past_overspend_usd": 1.5, "past_overspend_tokens": 500,
        "estimate_basis": "",
    })

    assert result["id"] == "exp-legacy"
    assert len(cost_apply.list_applied(cfg)) == 1  # no duplicate record created


def test_mark_applied_still_dedupes_new_format_exactly(cfg, db):
    """Two calls with the identical NEW-format signature remain idempotent
    (the ordinary case, unaffected by the legacy shim)."""
    proposal = {
        "signature": "cost:downsize:svc-a:anthropic:claude-opus-4-8:claude-haiku-4-5",
        "analyzer": "downsize", "title": "svc-a", "target_key": {},
        "agent_id": "", "baseline": {},
        "past_overspend_usd": 1.5, "past_overspend_tokens": 500,
        "estimate_basis": "",
    }
    first = cost_apply.mark_applied(db, cfg, proposal)
    second = cost_apply.mark_applied(db, cfg, proposal)
    assert first["id"] == second["id"]
    assert len(cost_apply.list_applied(cfg)) == 1


def test_a_reverted_legacy_record_does_not_shadow_a_fresh_apply(cfg, db):
    """A REVERTED legacy mark must not suppress a new application -- reverting
    means the user undid the change, so the model-qualified signature should
    be free to record a fresh apply."""
    reverted_legacy = {
        "id": "exp-legacy", "expectation_id": "exp-legacy",
        "signature": "cost:downsize:svc-a",
        "analyzer": "downsize", "kind": "cost", "title": "legacy",
        "target_key": {}, "agent_id": "", "applied_at": utcnow().isoformat(),
        "baseline": {}, "past_overspend_usd": 3.0, "past_overspend_tokens": 1000,
        "estimate_basis": "", "state": "reverted", "reverted_at": utcnow().isoformat(),
        "verify": {},
    }
    cost_apply._write_ledger(cfg, [reverted_legacy])

    result = cost_apply.mark_applied(db, cfg, {
        "signature": "cost:downsize:svc-a:anthropic:claude-opus-4-8:claude-haiku-4-5",
        "analyzer": "downsize", "title": "svc-a", "target_key": {},
        "agent_id": "", "baseline": {},
        "past_overspend_usd": 1.5, "past_overspend_tokens": 500,
        "estimate_basis": "",
    })

    assert result["id"] != "exp-legacy"  # a genuinely NEW record was created
    assert len(cost_apply.list_applied(cfg)) == 2
