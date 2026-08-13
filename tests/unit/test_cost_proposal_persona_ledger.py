"""The cost-proposal ledger has to be persona-RESOLVABLE, not persona-labelled.

``GET /relearn/cost-proposals`` feeds the Dashboard hero band, the
Total-opportunity tile and every Optimize sub-page's inline fix cards. Until the
ledger could be narrowed, all three published a whole-corpus dollar total under
whichever persona the reader had picked.

The narrowing cannot happen on read. ``/optimize`` gets to honour a persona
parameter by slicing a set of analyzer NAMES, and a set is separable; a dollar
total summed over a mixed corpus is not. So these tests pin the two halves of
the only design that can work: per-persona lists written at compute time, and an
explicit refusal to answer when a stored ledger has none.
"""
from __future__ import annotations

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import relearn_proposals, relearn_store


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _proposal(signature: str, usd: float) -> dict:
    return {
        "signature": signature,
        "kind": "cost:downsize",
        "title": signature,
        "past_overspend_usd": usd,
        "past_overspend_tokens": int(usd * 1000),
    }


def test_a_scoped_ledger_answers_each_persona_with_its_own_rows(cfg):
    relearn_store.write_cost_proposals(
        [_proposal("corpus", 100.0)],
        config=cfg,
        window_days=30,
        by_persona={
            "claude-code": [_proposal("cc", 90.0)],
            "sdk": [_proposal("sdk", 10.0)],
        },
    )

    cc, cc_ok = relearn_proposals.cost_proposals_scoped_to_persona(
        cfg, persona="claude-code",
    )
    sdk, sdk_ok = relearn_proposals.cost_proposals_scoped_to_persona(cfg, persona="sdk")

    assert cc_ok and sdk_ok
    assert [p["signature"] for p in cc] == ["cc"]
    assert [p["signature"] for p in sdk] == ["sdk"]
    # And the whole-corpus list is still what an unnarrowed read gets — it is
    # the honest answer to "everything", not a fallback for either persona.
    everything, ok = relearn_proposals.cost_proposals_scoped_to_persona(cfg)
    assert ok
    assert [p["signature"] for p in everything] == ["corpus"]


def test_personas_that_narrow_nothing_get_the_whole_corpus_list(cfg):
    """``mixed``/``unknown`` are not a third population.

    The classifier refuses to bucket those windows, so there is no scoped list
    to serve and none should be invented — the corpus list IS their answer, and
    they must read as RESOLVED rather than as an unanswerable request.
    """
    relearn_store.write_cost_proposals(
        [_proposal("corpus", 100.0)], config=cfg, window_days=30,
        by_persona={"claude-code": [_proposal("cc", 90.0)]},
    )
    for persona in ("mixed", "unknown"):
        rows, ok = relearn_proposals.cost_proposals_scoped_to_persona(
            cfg, persona=persona,
        )
        assert ok, persona
        assert [p["signature"] for p in rows] == ["corpus"], persona


def test_a_legacy_ledger_refuses_the_persona_rather_than_guessing(cfg):
    """The back-compat decision, stated as a test.

    A ledger written before per-persona proposals existed holds one
    whole-corpus list. Serving it under a persona label is the exact defect
    being fixed, and returning it as "this persona's empty result" is the same
    lie with a friendlier face. So: no rows, and ``resolved=False``.
    """
    relearn_store.write_cost_proposals(
        [_proposal("corpus", 100.0)], config=cfg, window_days=30,
    )
    block = relearn_store.read_cost_proposals(config=cfg)
    assert block["cost_persona_scoped"] is False
    assert block["cost_proposals_by_persona"] == {}

    rows, resolved = relearn_proposals.cost_proposals_scoped_to_persona(
        cfg, persona="sdk",
    )
    assert resolved is False
    assert rows == []
    # The unnarrowed read is unaffected — the legacy ledger is still perfectly
    # good at the question it can answer.
    rows, resolved = relearn_proposals.cost_proposals_scoped_to_persona(cfg)
    assert resolved is True
    assert len(rows) == 1


def test_an_unscoped_recompute_clears_a_previous_per_persona_block(cfg):
    """No stale scoped block beside a fresher corpus-wide list.

    That pairing is the torn artifact `cycle_provenance` exists to prevent: two
    measurements from two different moments, indistinguishable to a reader.
    Better to lose the narrowing and say so than to keep answering with it.
    """
    relearn_store.write_cost_proposals(
        [_proposal("corpus", 100.0)], config=cfg, window_days=30,
        by_persona={"sdk": [_proposal("sdk", 10.0)]},
    )
    assert relearn_store.read_cost_proposals(config=cfg)["cost_persona_scoped"] is True

    relearn_store.write_cost_proposals(
        [_proposal("corpus2", 120.0)], config=cfg, window_days=30,
    )
    block = relearn_store.read_cost_proposals(config=cfg)
    assert block["cost_persona_scoped"] is False
    assert block["cost_proposals_by_persona"] == {}
    _rows, resolved = relearn_proposals.cost_proposals_scoped_to_persona(
        cfg, persona="sdk",
    )
    assert resolved is False


def test_an_empty_scoped_list_is_not_an_unanswerable_one(cfg):
    """"This persona has nothing" and "we cannot say" are different states.

    A persona whose pass genuinely found no recoverable waste gets an empty
    list under ``resolved=True``, and a surface may render its empty state. The
    unscoped ledger above gets ``resolved=False`` and may not.
    """
    relearn_store.write_cost_proposals(
        [_proposal("corpus", 100.0)], config=cfg, window_days=30,
        by_persona={"claude-code": [_proposal("cc", 100.0)], "sdk": []},
    )
    rows, resolved = relearn_proposals.cost_proposals_scoped_to_persona(
        cfg, persona="sdk",
    )
    assert resolved is True
    assert rows == []


def test_each_persona_carries_its_own_relearn_finding(cfg):
    """Relearn reaches the headline through ``inbox_contribution``, not through
    the proposal list, so a scoped rollup needs a scoped finding or it folds the
    whole corpus's failure-recovery money onto one persona's proposals."""
    relearn_store.write_cost_proposals(
        [_proposal("corpus", 100.0)], config=cfg, window_days=30,
        by_persona={"claude-code": [], "sdk": []},
        relearn_by_persona={"claude-code": {"clusters": ["a"]}, "sdk": {"clusters": []}},
    )
    block = relearn_store.read_cost_proposals(config=cfg)
    assert block["cost_relearn_by_persona"]["claude-code"] == {"clusters": ["a"]}
    assert block["cost_relearn_by_persona"]["sdk"] == {"clusters": []}
