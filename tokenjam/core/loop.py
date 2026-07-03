"""Close-the-loop: annotations, expectations, and a fix-history ledger (#53).

The capture half of tokenjam (``tj trace`` / span timeline / auto model+backend
info) is strong, but the *loop-closing* half was entirely absent: a user could
see "something weird happened" but had no way to (a) leave a human note/verdict
on that run, (b) promote a bad run into a stored expectation/case, or (c) rerun
after a change and keep a history of what fixed it or made it worse.

This module is that missing half, deliberately **local-first** — no eval
platform, no cloud, no Langfuse round-trip (the Langfuse integration is *inbound*
ingestion only; see ``core/ingest_adapters/langfuse.py``). It is a thin ledger
over sessions the user already captures, in the same spirit as the drift
baselines and the method snapshots that preserve "how an agent attempted" work.

Product scope is deliberately small (YAGNI): tokenjam owns the *loop primitive*
(note → expectation → pass/regress history), NOT a full assertion/eval runner.
Judging pass vs. regress stays a **human verdict** — consistent with the
honesty discipline elsewhere (Critical Rule 14: we describe, we don't grade).
See ``docs/internal/close-the-loop.md`` for the full decision record.

Pure domain logic: no imports from ``tokenjam.cli`` / ``tokenjam.api`` (package
rule). Storage helpers accept either a ``StorageBackend`` (whose per-thread
``.conn`` cursor is used) or a raw DuckDB connection, mirroring
``db.set_session_label``. Parameterised SQL only (Critical Rule 7);
timestamps via ``utcnow()`` (Critical Rule 9).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from tokenjam.utils.time_parse import utcnow

# A run's after-the-fact verdict. `unknown` is the explicit "I looked but can't
# yet say" state; a bare note carries verdict=None.
VALID_VERDICTS = frozenset({"good", "bad", "mixed", "unknown"})
# The outcome of a rerun measured against an expectation. Human judgment, not an
# automated assertion result (see module docstring).
VALID_OUTCOMES = frozenset({"pass", "regress", "unknown"})

# Bound free-text so a pathological paste can't bloat the local DB.
MAX_NOTE_LEN = 4000
MAX_NAME_LEN = 200
MAX_DESC_LEN = 8000


@dataclass(frozen=True)
class RunAnnotation:
    """A human note + optional verdict left on a run (session), after the fact."""

    annotation_id: str
    session_id: str
    verdict: str | None
    note: str
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "annotation_id": self.annotation_id,
            "session_id": self.session_id,
            "verdict": self.verdict,
            "note": self.note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass(frozen=True)
class Expectation:
    """A run promoted into a stored expectation/case to check reruns against."""

    expectation_id: str
    origin_session_id: str | None
    agent_id: str | None
    name: str
    description: str | None
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "expectation_id": self.expectation_id,
            "origin_session_id": self.origin_session_id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass(frozen=True)
class ExpectationRun:
    """One fix-history ledger entry: a rerun's outcome against an expectation."""

    run_ledger_id: str
    expectation_id: str
    session_id: str | None
    outcome: str
    note: str | None
    created_at: datetime

    def to_dict(self) -> dict:
        return {
            "run_ledger_id": self.run_ledger_id,
            "expectation_id": self.expectation_id,
            "session_id": self.session_id,
            "outcome": self.outcome,
            "note": self.note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def _resolve_conn(db_or_conn):
    """Underlying cursor for a backend, or the conn passed as-is (see db.py)."""
    return getattr(db_or_conn, "conn", db_or_conn)


def _new_id() -> str:
    return uuid.uuid4().hex


def _clean(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped[:limit] if stripped else None


# --- Annotations -------------------------------------------------------------

def add_annotation(
    db_or_conn, session_id: str, *, note: str, verdict: str | None = None
) -> RunAnnotation:
    """Append a note + optional verdict to a run. Raises ValueError on bad input.

    ``note`` is required (an annotation with neither note nor verdict is
    meaningless); ``verdict`` is optional but, when given, must be a
    ``VALID_VERDICTS`` member. Append-only — multiple annotations per session are
    the norm (a running log), unlike the single-row ``session_labels`` rename.
    """
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    clean_note = _clean(note, MAX_NOTE_LEN)
    if not clean_note:
        raise ValueError("note is required")
    if verdict is not None:
        verdict = verdict.strip().lower()
        if verdict not in VALID_VERDICTS:
            raise ValueError(
                f"verdict must be one of {sorted(VALID_VERDICTS)}, got {verdict!r}"
            )
    conn = _resolve_conn(db_or_conn)
    ann = RunAnnotation(
        annotation_id=_new_id(),
        session_id=sid,
        verdict=verdict,
        note=clean_note,
        created_at=utcnow(),
    )
    conn.execute(
        "INSERT INTO run_annotations "
        "(annotation_id, session_id, verdict, note, created_at) "
        "VALUES ($1, $2, $3, $4, $5)",
        [ann.annotation_id, ann.session_id, ann.verdict, ann.note, ann.created_at],
    )
    return ann


def list_annotations(db_or_conn, session_id: str) -> list[RunAnnotation]:
    """Every annotation on a run, newest first."""
    conn = _resolve_conn(db_or_conn)
    if conn is None:
        return []
    rows = conn.execute(
        "SELECT annotation_id, session_id, verdict, note, created_at "
        "FROM run_annotations WHERE session_id = $1 ORDER BY created_at DESC",
        [session_id],
    ).fetchall()
    return [
        RunAnnotation(
            annotation_id=r[0], session_id=r[1], verdict=r[2],
            note=r[3], created_at=r[4],
        )
        for r in rows
    ]


# --- Expectations ------------------------------------------------------------

def create_expectation(
    db_or_conn,
    *,
    name: str,
    description: str | None = None,
    origin_session_id: str | None = None,
    agent_id: str | None = None,
) -> Expectation:
    """Promote a labeled run into a stored expectation/case.

    ``name`` is required. ``origin_session_id`` records the run it was promoted
    FROM (nullable — an expectation may be authored free-standing). Raises
    ValueError when ``name`` is empty.
    """
    clean_name = _clean(name, MAX_NAME_LEN)
    if not clean_name:
        raise ValueError("name is required")
    conn = _resolve_conn(db_or_conn)
    exp = Expectation(
        expectation_id=_new_id(),
        origin_session_id=(origin_session_id or None),
        agent_id=(agent_id or None),
        name=clean_name,
        description=_clean(description, MAX_DESC_LEN),
        created_at=utcnow(),
    )
    conn.execute(
        "INSERT INTO expectations "
        "(expectation_id, origin_session_id, agent_id, name, description, created_at) "
        "VALUES ($1, $2, $3, $4, $5, $6)",
        [
            exp.expectation_id, exp.origin_session_id, exp.agent_id,
            exp.name, exp.description, exp.created_at,
        ],
    )
    return exp


def _row_to_expectation(r) -> Expectation:
    return Expectation(
        expectation_id=r[0], origin_session_id=r[1], agent_id=r[2],
        name=r[3], description=r[4], created_at=r[5],
    )


_EXPECTATION_COLS = (
    "expectation_id, origin_session_id, agent_id, name, description, created_at"
)


def list_expectations(db_or_conn) -> list[Expectation]:
    """All expectations, newest first."""
    conn = _resolve_conn(db_or_conn)
    if conn is None:
        return []
    rows = conn.execute(
        f"SELECT {_EXPECTATION_COLS} FROM expectations ORDER BY created_at DESC"
    ).fetchall()
    return [_row_to_expectation(r) for r in rows]


def get_expectation(db_or_conn, expectation_id: str) -> Expectation | None:
    """A single expectation by id, or None."""
    conn = _resolve_conn(db_or_conn)
    if conn is None:
        return None
    row = conn.execute(
        f"SELECT {_EXPECTATION_COLS} FROM expectations WHERE expectation_id = $1",
        [expectation_id],
    ).fetchone()
    return _row_to_expectation(row) if row else None


def expectations_for_session(db_or_conn, session_id: str) -> list[Expectation]:
    """Expectations promoted FROM this session, newest first."""
    conn = _resolve_conn(db_or_conn)
    if conn is None:
        return []
    rows = conn.execute(
        f"SELECT {_EXPECTATION_COLS} FROM expectations "
        "WHERE origin_session_id = $1 ORDER BY created_at DESC",
        [session_id],
    ).fetchall()
    return [_row_to_expectation(r) for r in rows]


# --- Fix-history ledger ------------------------------------------------------

def record_expectation_run(
    db_or_conn,
    expectation_id: str,
    *,
    outcome: str,
    session_id: str | None = None,
    note: str | None = None,
) -> ExpectationRun:
    """Record a rerun's outcome against an expectation (the fix-history ledger).

    ``outcome`` must be a ``VALID_OUTCOMES`` member. Raises ValueError on a bad
    outcome or an unknown ``expectation_id`` (so a typo doesn't silently write an
    orphan ledger row).
    """
    outcome = (outcome or "").strip().lower()
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"outcome must be one of {sorted(VALID_OUTCOMES)}, got {outcome!r}"
        )
    if get_expectation(db_or_conn, expectation_id) is None:
        raise ValueError(f"unknown expectation_id {expectation_id!r}")
    conn = _resolve_conn(db_or_conn)
    entry = ExpectationRun(
        run_ledger_id=_new_id(),
        expectation_id=expectation_id,
        session_id=(session_id or None),
        outcome=outcome,
        note=_clean(note, MAX_NOTE_LEN),
        created_at=utcnow(),
    )
    conn.execute(
        "INSERT INTO expectation_runs "
        "(run_ledger_id, expectation_id, session_id, outcome, note, created_at) "
        "VALUES ($1, $2, $3, $4, $5, $6)",
        [
            entry.run_ledger_id, entry.expectation_id, entry.session_id,
            entry.outcome, entry.note, entry.created_at,
        ],
    )
    return entry


def list_expectation_runs(db_or_conn, expectation_id: str) -> list[ExpectationRun]:
    """The fix-history for an expectation, newest first."""
    conn = _resolve_conn(db_or_conn)
    if conn is None:
        return []
    rows = conn.execute(
        "SELECT run_ledger_id, expectation_id, session_id, outcome, note, created_at "
        "FROM expectation_runs WHERE expectation_id = $1 ORDER BY created_at DESC",
        [expectation_id],
    ).fetchall()
    return [
        ExpectationRun(
            run_ledger_id=r[0], expectation_id=r[1], session_id=r[2],
            outcome=r[3], note=r[4], created_at=r[5],
        )
        for r in rows
    ]
