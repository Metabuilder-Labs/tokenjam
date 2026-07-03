"""Tests for the close-the-loop primitive (#53): the core.loop storage helpers
(annotations, expectations, fix-history ledger) round-trip + validation.

Local-first: annotate a run -> promote it into an expectation -> record whether
reruns pass or regress. See core/loop.py.
"""
from __future__ import annotations

import pytest

from tokenjam.core import loop
from tokenjam.core.db import InMemoryBackend


# ── Annotations ────────────────────────────────────────────────────────────

def test_annotation_round_trip_newest_first():
    db = InMemoryBackend()
    try:
        assert loop.list_annotations(db, "sid-1") == []

        a1 = loop.add_annotation(db, "sid-1", note="looked wrong", verdict="bad")
        a2 = loop.add_annotation(db, "sid-1", note="still wrong after fix")
        # Many annotations per session (append-only log, not an upsert).
        rows = loop.list_annotations(db, "sid-1")
        assert [r.annotation_id for r in rows] == [a2.annotation_id, a1.annotation_id]
        assert rows[1].verdict == "bad"
        assert rows[0].verdict is None            # bare note carries no verdict
        # Isolation: a different session sees none of these.
        assert loop.list_annotations(db, "sid-2") == []
    finally:
        db.close()


def test_annotation_requires_note():
    db = InMemoryBackend()
    try:
        with pytest.raises(ValueError):
            loop.add_annotation(db, "sid-1", note="   ")
    finally:
        db.close()


def test_annotation_rejects_bad_verdict():
    db = InMemoryBackend()
    try:
        with pytest.raises(ValueError):
            loop.add_annotation(db, "sid-1", note="x", verdict="terrible")
    finally:
        db.close()


def test_annotation_note_truncated():
    db = InMemoryBackend()
    try:
        a = loop.add_annotation(db, "sid-1", note="y" * (loop.MAX_NOTE_LEN + 50))
        assert len(a.note) == loop.MAX_NOTE_LEN
    finally:
        db.close()


# ── Expectations ───────────────────────────────────────────────────────────

def test_expectation_create_and_list():
    db = InMemoryBackend()
    try:
        assert loop.list_expectations(db) == []
        e = loop.create_expectation(
            db, name="no infinite retry loop",
            description="should not retry the same tool 4x",
            origin_session_id="sid-1", agent_id="claude-code",
        )
        got = loop.get_expectation(db, e.expectation_id)
        assert got is not None
        assert got.name == "no infinite retry loop"
        assert got.origin_session_id == "sid-1"
        assert [x.expectation_id for x in loop.list_expectations(db)] == [e.expectation_id]
        # Scoped-by-origin lookup for the Lens "expectations from this run" list.
        assert [x.expectation_id for x in loop.expectations_for_session(db, "sid-1")] == [
            e.expectation_id
        ]
        assert loop.expectations_for_session(db, "other") == []
    finally:
        db.close()


def test_expectation_requires_name():
    db = InMemoryBackend()
    try:
        with pytest.raises(ValueError):
            loop.create_expectation(db, name="  ")
    finally:
        db.close()


def test_get_unknown_expectation_returns_none():
    db = InMemoryBackend()
    try:
        assert loop.get_expectation(db, "does-not-exist") is None
    finally:
        db.close()


# ── Fix-history ledger ─────────────────────────────────────────────────────

def test_ledger_records_pass_and_regress_newest_first():
    db = InMemoryBackend()
    try:
        e = loop.create_expectation(db, name="case", origin_session_id="sid-1")
        assert loop.list_expectation_runs(db, e.expectation_id) == []

        loop.record_expectation_run(
            db, e.expectation_id, outcome="regress", session_id="sid-2",
            note="still broken",
        )
        r2 = loop.record_expectation_run(
            db, e.expectation_id, outcome="pass", session_id="sid-3",
            note="fixed by prompt change",
        )
        runs = loop.list_expectation_runs(db, e.expectation_id)
        assert [r.outcome for r in runs] == ["pass", "regress"]  # newest first
        assert runs[0].run_ledger_id == r2.run_ledger_id
        assert runs[0].session_id == "sid-3"
    finally:
        db.close()


def test_ledger_rejects_bad_outcome():
    db = InMemoryBackend()
    try:
        e = loop.create_expectation(db, name="case")
        with pytest.raises(ValueError):
            loop.record_expectation_run(db, e.expectation_id, outcome="maybe")
    finally:
        db.close()


def test_ledger_rejects_unknown_expectation():
    db = InMemoryBackend()
    try:
        with pytest.raises(ValueError, match="unknown expectation_id"):
            loop.record_expectation_run(db, "ghost", outcome="pass")
    finally:
        db.close()


def test_ledger_isolated_per_expectation():
    db = InMemoryBackend()
    try:
        e1 = loop.create_expectation(db, name="one")
        e2 = loop.create_expectation(db, name="two")
        loop.record_expectation_run(db, e1.expectation_id, outcome="pass")
        assert len(loop.list_expectation_runs(db, e1.expectation_id)) == 1
        assert loop.list_expectation_runs(db, e2.expectation_id) == []
    finally:
        db.close()
