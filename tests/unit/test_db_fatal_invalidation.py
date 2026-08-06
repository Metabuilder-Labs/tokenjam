"""A fatal DuckDB error must stop the process serving, not be logged per-row.

The failure these pin, observed on a real database: a background analyzer pass
persisting one agent-config row raised

    duckdb.FatalException: FATAL Error: Invalid Input Error: Failed to delete
    all rows from index. Only deleted 0 out of 1 rows.

which `DuckDBAgentConfigStore` caught and logged as a soft "this record could
not be persisted" warning before carrying on. But that exception invalidates
the whole DuckDB database INSTANCE, not the connection that raised it — so
every later query in the process, on every connection including ones opened
afterwards, failed with "database has been invalidated because of a previous
fatal error". Every API route 500'd, the dashboard rendered empty, and
`/health` went on returning `{"status": "ok"}` because it never touched the
database.

Three properties, each of which alone would have prevented the outage:

  * the per-record handler must not swallow a fatal (`test_fatal_*`);
  * the health probe must ask the database rather than assert liveness
    (`test_health_*`);
  * recovery must close EVERY connection before reopening, because DuckDB
    hands back the same invalidated instance otherwise (`test_recover_*`).

The root-cause index fault is a persistent property of one database file, so
the fatal itself is simulated here; `test_repair_agent_config_index_*` pins the
repair that clears it.
"""
from __future__ import annotations

import time

import duckdb
import pytest
from fastapi.testclient import TestClient

from tokenjam.core.agent_config import ConfigRecord, DuckDBAgentConfigStore
from tokenjam.core.db import (
    DuckDBBackend,
    check_index_divergence,
    explicit_indexes,
    clear_fatal_db_error,
    fatal_db_error,
    is_fatal_db_error,
    note_fatal_db_error,
    recover_invalidated_database,
    repair_explicit_indexes,
    run_migrations,
)
from tokenjam.core.config import StorageConfig
from tokenjam.utils.time_parse import utcnow


@pytest.fixture(autouse=True)
def _clean_fatal_state():
    """The fatal record is process-wide (the invalidation is), so reset it."""
    clear_fatal_db_error()
    yield
    clear_fatal_db_error()


@pytest.fixture
def backend(tmp_path):
    b = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield b
    b.close()


class _StubConfig:
    """Stands in for TjConfig: the cycle raises before it reads any of it."""


def _record(name: str = "CLAUDE.md") -> ConfigRecord:
    return ConfigRecord(
        kind="instruction",
        scope="project",
        root="/tmp/p",
        name=name,
        path=f"/tmp/p/{name}",
        size_bytes=10,
        tokens=3,
        content_hash="h",
        last_seen=utcnow(),
    )


# --- classification --------------------------------------------------------

def test_fatal_exception_is_classified_fatal():
    assert is_fatal_db_error(duckdb.FatalException("FATAL Error: boom"))


def test_invalidation_message_is_classified_fatal_even_if_rewrapped():
    # The type is authoritative, but a fatal that reaches a handler wrapped by
    # an intermediate layer must still be recognised — misclassifying it is
    # exactly what turns the outage silent.
    assert is_fatal_db_error(
        RuntimeError("database has been invalidated because of a previous fatal error")
    )


def test_ordinary_write_conflict_is_not_fatal():
    # The conflict-tolerant retry path must keep working: these are the errors
    # the agent-config store is DESIGNED to degrade on, and promoting them to
    # fatal would take down a pass for a recoverable race.
    assert not is_fatal_db_error(duckdb.ConstraintException("Duplicate key"))
    assert not is_fatal_db_error(duckdb.TransactionException("Conflict on tuple deletion!"))


# --- the per-record handler ------------------------------------------------

def test_fatal_during_upsert_is_raised_not_degraded(backend):
    """The regression. A fatal inside the persistence loop must escape it."""
    store = DuckDBAgentConfigStore(backend.conn, lock=backend.write_lock)

    class Exploding:
        def execute(self, *a, **k):
            raise duckdb.FatalException(
                "FATAL Error: Invalid Input Error: Failed to delete all rows "
                "from index. Only deleted 0 out of 1 rows."
            )

    store.conn = Exploding()
    with pytest.raises(duckdb.FatalException):
        store.upsert([_record()])


def test_fatal_during_upsert_is_recorded_process_wide(backend):
    """...and leaves a record, so surfaces stop claiming to be healthy."""
    store = DuckDBAgentConfigStore(backend.conn, lock=backend.write_lock)

    class Exploding:
        def execute(self, *a, **k):
            raise duckdb.FatalException("FATAL Error: Failed to delete all rows from index.")

    store.conn = Exploding()
    assert fatal_db_error() is None
    with pytest.raises(duckdb.FatalException):
        store.upsert([_record()])
    assert "Failed to delete all rows from index" in (fatal_db_error() or "")


def test_recoverable_write_failure_still_degrades_quietly(backend, caplog):
    """The inverse guard: widening 'fatal' must not break ordinary degrading.

    A write-write conflict is the case the store exists to absorb — it must
    still land in the mirror, still be readable, and still not raise.
    """
    store = DuckDBAgentConfigStore(backend.conn, lock=backend.write_lock)

    class Conflicted:
        def execute(self, *a, **k):
            raise duckdb.ConstraintException("Duplicate key")

    store.conn = Conflicted()
    store.upsert([_record()])  # must not raise
    assert store.degraded
    assert fatal_db_error() is None
    assert [r.name for r in store.select()] == ["CLAUDE.md"]


# --- connection health and recovery ----------------------------------------

def test_check_health_true_on_a_live_backend(backend):
    assert backend.check_health() is True


def test_check_health_false_when_the_connection_cannot_answer(backend):
    backend.close()
    assert backend.check_health() is False


def test_recover_reestablishes_a_torn_down_backend(backend):
    """Closing every handle then reopening is the only in-process recovery.

    Verified against the engine: while ANY connection to the path survives,
    `duckdb.connect` returns the same (invalidated) instance from DuckDB's
    per-path cache, so a reconnect that does not close first recovers nothing.
    """
    backend.conn.execute(
        "INSERT INTO agent_config_files "
        "(config_id, kind, scope, path, last_seen) VALUES ('a','instruction','p','/x',now())"
    )
    backend._teardown_connections()
    assert backend.check_health() is False

    note_fatal_db_error(duckdb.FatalException("FATAL Error: simulated"))
    assert recover_invalidated_database() is True

    assert backend.check_health() is True
    assert fatal_db_error() is None
    row = backend.conn.execute("SELECT COUNT(*) FROM agent_config_files").fetchone()
    assert row[0] == 1, "recovery must reopen the database, not replace it"


def test_recover_hands_every_thread_a_fresh_cursor(backend):
    """A thread holding a pre-recovery cursor must not keep using it."""
    stale = backend.conn
    backend._teardown_connections()
    recover_invalidated_database()
    assert backend.conn is not stale
    backend.conn.execute("SELECT 1").fetchone()


def test_recovery_closes_cursors_belonging_to_other_threads(backend):
    """The reason cursors are tracked in a list rather than left to the thread.

    Request-path cursors live in a `threading.local`, which the recovering
    thread cannot enumerate. One surviving handle keeps the invalidated
    instance in DuckDB's per-path cache, so a recovery that closed only its own
    connection would reopen straight back onto the dead database. After
    recovery every thread — including ones that were holding a cursor — must be
    able to query again.
    """
    import threading as _t

    errors: list[str] = []
    started, resume = _t.Event(), _t.Event()

    def worker():
        try:
            backend.conn.execute("SELECT 1").fetchone()  # take a per-thread cursor
            started.set()
            resume.wait(5)
            backend.conn.execute("SELECT 1").fetchone()  # must work post-recovery
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
            started.set()

    backend.conn.execute("SELECT 1").fetchone()  # this thread's cursor
    t = _t.Thread(target=worker)
    t.start()
    assert started.wait(5)
    assert len(backend._cursors) == 2, "both threads' cursors must be tracked"

    backend._teardown_connections()
    assert recover_invalidated_database() is True
    resume.set()
    t.join(10)

    assert errors == []
    assert backend._cursors, "recovery must hand out fresh cursors, not reuse closed ones"


def test_recovery_rebuilds_every_index_not_only_the_flagged_ones(backend, monkeypatch):
    """The correctness property that a cheaper recovery would break.

    The fatal does not name the index that raised it and the probe only
    compares sampled values, so recovering by repairing the probe's verdict can
    reconnect straight back into the same fatal. Measured on a real damaged
    database: a sampled sweep found three of four, and the next write still
    died. Recovery therefore rebuilds the lot.
    """
    import tokenjam.core.db as db_mod

    repaired: list = []
    monkeypatch.setattr(
        db_mod, "repair_explicit_indexes",
        lambda conn, names=None: repaired.append(names) or ["x"],
    )
    monkeypatch.setattr(
        db_mod, "check_index_divergence",
        lambda conn: ([("idx_agent_config_kind", "agent_config_files", "bad")], []),
    )
    backend._teardown_connections()
    assert recover_invalidated_database() is True
    assert repaired == [None], "recovery must not scope the repair to the faults"


def test_no_thread_gets_a_cursor_from_a_half_torn_down_backend(backend, monkeypatch):
    """The window between teardown and reopen must not be observable.

    Recovery closes every connection and only then reopens them. Between those
    two steps `conn` used to be reachable with the root connection already
    closed, so a request or ingest thread arriving mid-recovery got a cursor
    from a dead handle — a 500 or a dropped write caused BY the recovery. That
    would make this change do the very thing it exists to prevent.

    Driven under a genuinely widened window rather than by luck: `_reopen` is
    slowed so the racing thread is guaranteed to arrive inside it.
    """
    import threading as _t

    backend.conn.execute("SELECT 1").fetchone()
    real_reopen = type(backend)._reopen
    inside = _t.Event()

    def slow_reopen(self):
        inside.set()
        time.sleep(0.4)
        return real_reopen(self)

    monkeypatch.setattr(type(backend), "_reopen", slow_reopen)

    errors: list[str] = []
    results: list = []

    def racer():
        inside.wait(5)
        try:
            results.append(backend.conn.execute("SELECT 1").fetchone())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    t = _t.Thread(target=racer)
    t.start()
    assert recover_invalidated_database() is True
    t.join(10)

    assert errors == [], f"a thread was handed a dead connection: {errors}"
    assert results == [(1,)]


def test_in_memory_backend_is_not_torn_down_by_recovery():
    """Its database IS its connection — 'recovering' it would delete the data."""
    from tokenjam.core.db import InMemoryBackend

    mem = InMemoryBackend()
    mem.conn.execute(
        "INSERT INTO agent_config_files "
        "(config_id, kind, scope, path, last_seen) VALUES ('a','instruction','p','/x',now())"
    )
    assert mem.recoverable is False
    recover_invalidated_database()
    row = mem.conn.execute("SELECT COUNT(*) FROM agent_config_files").fetchone()
    assert row[0] == 1


# --- the broad `except Exception` handlers ---------------------------------
#
# Both background jobs are fire-and-forget: they start a daemon thread and
# return, and each thread already swallows every exception ("never crash a
# thread", "errors are logged, never raised"). That is right for a job that
# failed and catastrophic for a fatal, and it is where the outage's exception
# actually died — a guard around the DISPATCH would never have seen it.

def test_handle_if_fatal_recovers_and_reports_it_handled(backend):
    from tokenjam.core.db import handle_if_fatal

    backend._teardown_connections()
    assert handle_if_fatal(
        duckdb.FatalException("FATAL Error: simulated"), what="a job"
    ) is True
    assert backend.check_health() is True
    assert fatal_db_error() is None


def test_handle_if_fatal_leaves_ordinary_failures_to_their_caller(backend):
    """It must return False for anything else, or every job failure would
    trigger a needless full teardown of the process's connections."""
    from tokenjam.core.db import handle_if_fatal

    assert handle_if_fatal(ValueError("a parse failed"), what="a job") is False
    assert handle_if_fatal(
        duckdb.ConstraintException("Duplicate key"), what="a job"
    ) is False
    assert backend.check_health() is True


def test_transcript_catch_up_does_not_swallow_a_fatal(backend, monkeypatch):
    """The catch-up thread's handler must escalate a fatal, not log a warning."""
    from tokenjam.core import transcript_sync

    seen: list[str] = []
    monkeypatch.setattr(
        "tokenjam.core.db.handle_if_fatal",
        lambda exc, what: (seen.append(what), True)[1],
    )
    monkeypatch.setattr(
        transcript_sync, "run_catch_up",
        lambda *a, **k: (_ for _ in ()).throw(
            duckdb.FatalException("FATAL Error: Failed to delete all rows from index.")
        ),
    )
    transcript_sync.start_catch_up(lambda: backend).join(10)
    assert seen == ["transcript catch-up"]


def test_scan_cycle_does_not_swallow_a_fatal(backend, monkeypatch):
    """Same guard on the analyzer cycle's `except Exception` thread handler."""
    from tokenjam.core.optimize import scan_cycle

    seen: list[str] = []
    monkeypatch.setattr(
        "tokenjam.core.db.handle_if_fatal",
        lambda exc, what: (seen.append(what), True)[1],
    )
    # Raise the fatal from the first thing the cycle's thread body does with
    # the database, so the real `except Exception` handler is what catches it.
    monkeypatch.setattr(
        "tokenjam.core.optimize.cycle_provenance.begin_cycle",
        lambda *a, **k: (_ for _ in ()).throw(
            duckdb.FatalException("FATAL Error: Failed to delete all rows from index.")
        ),
    )
    assert scan_cycle.trigger_scan_cycle(lambda: backend, _StubConfig(), force=True)
    for _ in range(100):
        if seen:
            break
        time.sleep(0.05)
    assert seen == ["analyzer scan cycle"]


def test_recovery_runs_even_when_every_handler_swallowed_the_fatal(backend):
    """The swallow-proof backstop.

    A fatal from an analyzer's write crosses several broad `except Exception`
    handlers before any of ours — the per-analyzer one in the optimize runner
    records it and continues with the rest, so the exception never reaches the
    scan cycle's handler at all. Recovery therefore cannot depend on catching
    it: `note_fatal_db_error` records the fatal where it is RECOGNISED, and
    this keys off that record.
    """
    from tokenjam.core.db import recover_if_fatal_noted

    backend._teardown_connections()
    note_fatal_db_error(duckdb.FatalException("FATAL Error: simulated"))
    # Nobody re-raised; the exception was absorbed. Recovery must still happen.
    assert recover_if_fatal_noted(what="a pass") is True
    assert backend.check_health() is True
    assert fatal_db_error() is None


def test_backstop_is_a_no_op_when_nothing_was_recorded(backend):
    from tokenjam.core.db import recover_if_fatal_noted

    assert recover_if_fatal_noted(what="a pass") is False


def test_optimize_runner_does_not_absorb_a_fatal_as_one_analyzer_failing(backend, monkeypatch):
    """The finding the review bot raised, driven through the REAL runner.

    The runner isolates each analyzer so one failure cannot lose the others.
    For a fatal that isolation is wrong twice over: every analyzer after it
    runs against a dead database, and the report ends up carrying a "did not
    complete" note per analyzer for a reason that has nothing to do with them.
    Worse, absorbing it here is why the fatal never reached the scan cycle's
    handler, so nothing recovered the connection.

    Asserted by BEHAVIOUR: `build_report` must propagate, and must not go on to
    run the analyzers queued behind the one that died.
    """
    from datetime import timedelta

    from tokenjam.core.optimize import runner
    from tokenjam.core.config import TjConfig

    ran: list[str] = []

    def fatal_analyzer(_ctx):
        ran.append("fatal-one")
        raise duckdb.FatalException(
            "FATAL Error: Invalid Input Error: Failed to delete all rows from index."
        )

    def innocent_analyzer(_ctx):
        ran.append("innocent")

    monkeypatch.setattr(
        runner, "ANALYZER_REGISTRY",
        {"fatal-one": fatal_analyzer, "innocent": innocent_analyzer},
    )
    monkeypatch.setattr(runner, "ANALYZER_ORDER", ["fatal-one", "innocent"])

    with pytest.raises(duckdb.FatalException):
        runner.build_report(
            backend, TjConfig(version="1"),
            since=utcnow() - timedelta(days=1), until=utcnow(),
        )

    assert ran == ["fatal-one"], (
        "the runner must stop on a fatal, not keep running analyzers against a "
        f"database that is already dead (ran: {ran})"
    )


# --- the health surface ----------------------------------------------------

def _app(db):
    from tokenjam.api.app import create_app
    from tokenjam.core.config import ApiAuthConfig, ApiConfig, TjConfig

    config = TjConfig(version="1", api=ApiConfig(auth=ApiAuthConfig(enabled=False)))
    return create_app(config, db, ingest_pipeline=object())


def test_health_reports_ok_and_says_storage_was_checked(backend):
    with TestClient(_app(backend)) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["storage"] == "ok"


def test_health_reports_503_when_the_database_cannot_be_recovered(backend, monkeypatch):
    """The core surface regression: liveness must not be reported as health.

    A process that is up but cannot read its database served every route as a
    500 while `/health` returned `{"status": "ok"}`. Nothing may render a
    green status off a probe that never asked the database.
    """
    monkeypatch.setattr(type(backend), "check_health", lambda self: False)
    monkeypatch.setattr(
        "tokenjam.core.db.recover_invalidated_database", lambda **kw: False
    )
    with TestClient(_app(backend)) as client:
        resp = client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unhealthy"
    assert body["storage"] == "invalidated"


def test_health_recovers_an_invalidated_database_in_place(backend):
    """The polling that notices the outage is allowed to end it."""
    backend._teardown_connections()
    note_fatal_db_error(duckdb.FatalException("FATAL Error: simulated"))
    with TestClient(_app(backend)) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["storage"] == "recovered"
    assert "simulated" in body["recovered_from"]
    assert backend.check_health() is True


def test_health_never_claims_storage_state_it_did_not_check():
    """A backend that cannot be probed reports 'unknown', never 'ok'."""
    class Opaque:
        pass

    with TestClient(_app(Opaque())) as client:
        body = client.get("/health").json()
    assert body["storage"] == "unknown"


# --- the repair ------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = duckdb.connect(str(tmp_path / "r.duckdb"))
    run_migrations(c)
    yield c
    c.close()


def _seed(conn, n: int) -> None:
    for i in range(n):
        conn.execute(
            "INSERT INTO agent_config_files "
            "(config_id, kind, scope, path, last_seen, tokens) "
            "VALUES ($1,'instruction','project',$2, now(), $3)",
            [f"id-{i}", f"/tmp/p{i}/CLAUDE.md", i],
        )


def _index_names(conn) -> set:
    return {
        r[0] for r in conn.execute(
            "SELECT index_name FROM duckdb_indexes()"
        ).fetchall()
    }


def test_repair_rebuilds_every_explicit_index_by_default(conn):
    _seed(conn, 25)
    before = _index_names(conn)
    rebuilt = repair_explicit_indexes(conn)
    assert set(rebuilt) == before
    assert _index_names(conn) == before


def test_repair_can_be_limited_to_named_indexes(conn):
    """Recovery and doctor both repair only what diverged — rebuilding a sound
    index is correct but pointlessly expensive on a large table."""
    _seed(conn, 5)
    rebuilt = repair_explicit_indexes(conn, ["idx_agent_config_kind"])
    assert rebuilt == ["idx_agent_config_kind"]
    assert "idx_agent_config_last_seen" in _index_names(conn)


def test_repair_preserves_every_row(conn):
    _seed(conn, 25)
    repair_explicit_indexes(conn)
    rows = conn.execute(
        "SELECT config_id, tokens FROM agent_config_files ORDER BY tokens"
    ).fetchall()
    assert rows == [(f"id-{i}", i) for i in range(25)]


def test_repair_is_idempotent(conn):
    _seed(conn, 5)
    first = repair_explicit_indexes(conn)
    assert repair_explicit_indexes(conn) == first
    assert repair_explicit_indexes(conn) == first


def test_repair_is_safe_on_empty_tables(conn):
    assert set(repair_explicit_indexes(conn)) == _index_names(conn)


def test_repair_never_touches_a_primary_key(conn):
    """The safety property that makes the sweep runnable unattended.

    `duckdb_indexes()` does not list a PRIMARY KEY's own ART, so a repair
    driven by that catalogue structurally cannot drop one. Asserted by
    behaviour, not by inspection: the constraint still rejects a duplicate.
    """
    _seed(conn, 3)
    repair_explicit_indexes(conn)
    with pytest.raises(duckdb.ConstraintException):
        conn.execute(
            "INSERT INTO agent_config_files "
            "(config_id, kind, scope, path, last_seen) "
            "VALUES ('id-0','instruction','project','/dup', now())"
        )


def test_repair_leaves_the_table_writable(conn):
    """DELETE and INSERT OR REPLACE are the two statements a damaged index
    makes fatal, so a repair that did not restore them would clear nothing."""
    _seed(conn, 4)
    repair_explicit_indexes(conn)
    conn.execute("DELETE FROM agent_config_files WHERE config_id = 'id-1'")
    conn.execute(
        "INSERT OR REPLACE INTO agent_config_files "
        "(config_id, kind, scope, path, last_seen, tokens) "
        "VALUES ('id-2','instruction','project','/moved', now(), 99)"
    )
    rows = conn.execute(
        "SELECT config_id, tokens FROM agent_config_files ORDER BY config_id"
    ).fetchall()
    assert rows == [("id-0", 0), ("id-2", 99), ("id-3", 3)]


def test_repair_leaves_an_index_it_cannot_recreate_alone(conn, monkeypatch):
    """Dropping an index whose DDL is unknown would turn damage into absence,
    and nothing would put it back — migrations are already recorded applied."""
    _seed(conn, 3)
    import tokenjam.core.db as db_mod

    real = db_mod.explicit_indexes
    monkeypatch.setattr(db_mod, "explicit_indexes", lambda c: [
        (n, t, cols, "" if n == "idx_agent_config_kind" else ddl)
        for n, t, cols, ddl in real(c)
    ])
    rebuilt = repair_explicit_indexes(conn)
    assert "idx_agent_config_kind" not in rebuilt
    assert "idx_agent_config_kind" in _index_names(conn)


# --- the sweep -------------------------------------------------------------

def test_sweep_covers_every_explicit_index(conn):
    """Whatever we do not probe, we may not call sound — so the sweep must
    reach every index in the catalogue, not one table's worth."""
    _seed(conn, 10)
    tables = {t for _n, t, _c, _d in explicit_indexes(conn)}
    assert "agent_config_files" in tables and "spans" in tables and "sessions" in tables


def test_sweep_reports_no_faults_on_a_healthy_database(conn):
    _seed(conn, 10)
    faults, _unprobed = check_index_divergence(conn)
    assert faults == []


def test_sweep_is_quiet_on_empty_tables(conn):
    """Nothing to compare is not evidence of damage."""
    faults, unprobed = check_index_divergence(conn)
    assert faults == [] and unprobed == []


def test_sweep_excludes_primary_keys_from_what_it_reports(conn):
    _seed(conn, 4)
    assert all(
        not name.upper().startswith("PRIMARY")
        for name, _t, _c, _d in explicit_indexes(conn)
    )


def test_sweep_reports_an_unprobeable_index_rather_than_passing_it(conn):
    """A multi-column index cannot be tested by the single-column comparison.
    It must surface as not-proven, never be counted as sound."""
    _seed(conn, 4)
    conn.execute(
        "CREATE INDEX idx_acf_multi ON agent_config_files(kind, scope)"
    )
    faults, unproven = check_index_divergence(conn)
    assert faults == []
    assert any(name == "idx_acf_multi" for name, _t, _r in unproven)


def test_sweep_says_so_when_coverage_was_only_a_sample(conn, monkeypatch):
    """The property that stops a clean verdict being read as a guarantee.

    Learned by getting it wrong on a real damaged database: a three-value
    sample called an index clean, a repair trusted that verdict, and the very
    next write still raised the fatal. Partial coverage must be reported, not
    rounded up to "sound".
    """
    import tokenjam.core.db as db_mod

    monkeypatch.setattr(db_mod, "_PROBE_VALUE_LIMIT", 1)
    monkeypatch.setattr(db_mod, "_PROBE_SAMPLE_VALUES", 1)
    _seed(conn, 6)  # 6 distinct paths, 1 compared
    faults, unproven = check_index_divergence(conn)
    assert faults == []
    reasons = {name: reason for name, _t, reason in unproven}
    assert "idx_agent_config_last_seen" in reasons
    assert "not proof" in reasons["idx_agent_config_last_seen"]


def test_sweep_proves_soundness_when_every_value_was_compared(conn):
    """The inverse: full coverage on a small table earns an empty `unproven`,
    otherwise doctor could never report a clean bill of health at all."""
    _seed(conn, 6)
    faults, unproven = check_index_divergence(conn)
    assert faults == []
    acf = [name for name, table, _r in unproven if table == "agent_config_files"]
    assert acf == []


def test_index_columns_parses_the_catalogue_rendering():
    """`expressions` is a VARCHAR rendering of a list, not a list."""
    from tokenjam.core.db import _index_columns

    assert _index_columns("[kind]") == ["kind"]
    assert _index_columns("[agent_id, started_at]") == ["agent_id", "started_at"]
    assert _index_columns("") == []


def test_probe_uses_a_form_the_index_cannot_serve(conn):
    """The subtle half, and the reason this probe nearly did not work.

    `CAST(col AS VARCHAR)` is a NO-OP on a column already VARCHAR, so the
    planner discards it and the index serves both sides of the comparison — a
    probe built on it compares a damaged index against itself and reports sound
    whatever the damage. This pins that the scan form still agrees with a
    GROUP BY, which no index can serve, so the two sides genuinely differ.
    """
    _seed(conn, 12)
    truth = dict(conn.execute(
        "SELECT kind, COUNT(*) FROM agent_config_files GROUP BY 1"
    ).fetchall())
    scanned = conn.execute(
        "SELECT COUNT(*) FROM agent_config_files "
        "WHERE CAST(kind AS VARCHAR) || '' = CAST('instruction' AS VARCHAR)"
    ).fetchone()[0]
    assert scanned == truth["instruction"] == 12
