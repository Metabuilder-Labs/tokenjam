"""A record with no observed time is REJECTED, never repaired.

Ingest had two ways of inventing a timestamp it never saw, and both were worse
than declining the record.

`1970-01-01`, from a zero epoch, is not a neutral placeholder: it participates
in `MIN()`, in `ORDER BY`, and in every day union, so ONE such row made a corpus
with two months of usable history report a span in the thousands of days.
`datetime.now()`, the other substitute, is the harder one to catch — a 1970 row
looks wrong on sight, whereas a row dated today looks like an observation, and it
silently moved months-old spend into the present window on every backfill run.

The columns stay `NOT NULL`. A nullable column would move the problem into the
~25 `ORDER BY start_time` / `ORDER BY started_at` sites whose null placement is a
DuckDB session setting rather than a property of the query — `get_active_session`
and `get_session_by_conversation` among them, where a null-started row winning
`ORDER BY ... DESC LIMIT 1` would mis-attribute every subsequent span to it — and
it would buy only the ability to retain rows that can never time anything.

So the fix is at the boundary, and it is the idiom the ingest adapters already
used (`ingest_adapters/langfuse.py`, `helicone.py`): decline, and count the
decline. Rejections are visible on every path here; a fabricated timestamp never
was.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tokenjam.api.routes.logs import (
    _observed_timestamp_ns,
    _ts_to_datetime,
    parse_log_records,
)
from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.data_span import MIN_PLAUSIBLE_YEAR
from tokenjam.core.db import (
    DuckDBBackend,
    InMemoryBackend,
    count_sentinel_timestamp_rows,
    purge_sentinel_timestamp_rows,
)
from tokenjam.core.ingest import IngestPipeline, SpanRejectedError
from tokenjam.otel.otlp_parsing import parse_otlp_span
from tokenjam.otel.semconv import ClaudeCodeEvents, CodexEvents


# --- the logs path (Claude Code + Codex) ------------------------------------


def test_a_real_timestamp_is_taken_as_given():
    assert _observed_timestamp_ns({"timeUnixNano": "1800000000000000000"}, {}) == \
        1_800_000_000_000_000_000
    assert _ts_to_datetime(1_800_000_000_000_000_000) == datetime.fromtimestamp(
        1_800_000_000, tz=timezone.utc,
    )


def test_the_codex_iso_fallback_is_still_honoured():
    """Codex sets timeUnixNano=0 and puts the real time in an attribute."""
    ns = _observed_timestamp_ns(
        {"timeUnixNano": "0"},
        {CodexEvents.EVENT_TIMESTAMP: "2026-07-28T12:00:00Z"},
    )
    assert datetime.fromtimestamp(ns / 1e9, tz=timezone.utc) == datetime(
        2026, 7, 28, 12, 0, tzinfo=timezone.utc,
    )


@pytest.mark.parametrize("record,attrs", [
    ({}, {}),                        # field absent entirely
    ({"timeUnixNano": "0"}, {}),     # present and zero, no fallback available
    ({"timeUnixNano": 0}, {}),
])
def test_a_record_with_no_observed_time_is_rejected(record, attrs):
    with pytest.raises(SpanRejectedError) as exc:
        _observed_timestamp_ns(record, attrs)
    assert "no observed timestamp" in str(exc.value)


@pytest.mark.parametrize("record,attrs", [
    ({"timeUnixNano": "not-a-number"}, {}),
    ({"timeUnixNano": "0"}, {CodexEvents.EVENT_TIMESTAMP: "not-a-date"}),
    ({"timeUnixNano": "0"}, {CodexEvents.EVENT_TIMESTAMP: 12345}),
])
def test_an_unparseable_timestamp_rejects_the_record_not_the_batch(record, attrs):
    """These reads used to sit OUTSIDE the per-record guard.

    A malformed `timeUnixNano` raised `ValueError` and a non-string
    `event.timestamp` raised `AttributeError` from `.rstrip`, either of which
    escaped the loop and failed the whole export with a 500 — one bad record
    discarding every good one in the same batch.
    """
    with pytest.raises(SpanRejectedError):
        _observed_timestamp_ns(record, attrs)


def _log_body(records: list[dict]) -> dict:
    return {"resourceLogs": [{
        "resource": {"attributes": []},
        "scopeLogs": [{"logRecords": records}],
    }]}


def _api_request_record(timestamp_ns, *, session_id: str = "sess-1") -> dict:
    return {
        "timeUnixNano": timestamp_ns,
        "body": {"stringValue": ClaudeCodeEvents.API_REQUEST},
        "attributes": [
            {"key": ClaudeCodeEvents.SESSION_ID, "value": {"stringValue": session_id}},
            {"key": ClaudeCodeEvents.DURATION_MS, "value": {"intValue": "1200"}},
            {"key": "model", "value": {"stringValue": "claude-sonnet-5"}},
            {"key": ClaudeCodeEvents.INPUT_TOKENS, "value": {"intValue": "100"}},
            {"key": ClaudeCodeEvents.OUTPUT_TOKENS, "value": {"intValue": "10"}},
        ],
    }


@pytest.fixture
def pipeline():
    db = InMemoryBackend()
    yield IngestPipeline(db=db, config=TjConfig(version="1")), db
    db.close()


def test_an_untimed_record_is_counted_as_a_rejection_not_silently_dropped(pipeline):
    """A record tj refuses is a change to what the corpus holds, so it is
    reported the same way an accepted one is."""
    pipe, db = pipeline
    ingested, rejections = parse_log_records(
        _log_body([_api_request_record("0")]), pipe,
    )
    assert ingested == 0
    assert len(rejections) == 1
    assert "no observed timestamp" in rejections[0]["reason"]
    assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 0


def test_one_bad_record_does_not_cost_the_good_ones_in_its_batch(pipeline):
    pipe, db = pipeline
    ingested, rejections = parse_log_records(
        _log_body([
            _api_request_record("not-a-number"),
            _api_request_record("1800000000000000000"),
            _api_request_record("0"),
        ]),
        pipe,
    )
    assert ingested == 1
    assert len(rejections) == 2
    stored = db.conn.execute("SELECT start_time FROM spans").fetchall()
    assert len(stored) == 1
    assert stored[0][0].year >= MIN_PLAUSIBLE_YEAR


def test_no_ingested_span_can_carry_a_sentinel_year(pipeline):
    """The property, stated over the store rather than over one converter."""
    pipe, db = pipeline
    parse_log_records(
        _log_body([_api_request_record(ns) for ns in ("0", "1800000000000000000")]),
        pipe,
    )
    assert count_sentinel_timestamp_rows(db.conn) == {}


# --- the OTLP path ----------------------------------------------------------


def test_an_otlp_span_with_no_start_time_is_rejected():
    """Substituting `datetime.now()` dated historical work to whenever tj
    received it, which reads as a real observation."""
    with pytest.raises(SpanRejectedError) as exc:
        parse_otlp_span(
            {"spanId": "s1", "traceId": "t1", "name": "chat", "attributes": []}, {},
        )
    assert "no observed time" in str(exc.value)


def test_an_otlp_span_with_a_start_time_is_unaffected():
    span = parse_otlp_span(
        {
            "spanId": "s1", "traceId": "t1", "name": "chat",
            "startTimeUnixNano": "1800000000000000000",
            "endTimeUnixNano": "1800000001000000000",
            "attributes": [],
        },
        {},
    )
    assert span.start_time == datetime.fromtimestamp(1_800_000_000, tz=timezone.utc)
    assert span.duration_ms == pytest.approx(1000.0)


# --- the transcript backfill ------------------------------------------------


def _assistant_record(uuid_: str, ts: str | None, session_id: str = "sess-1") -> dict:
    record = {
        "type": "assistant", "uuid": uuid_, "sessionId": session_id, "cwd": "/repo",
        "message": {
            "id": f"msg-{uuid_}", "model": "claude-sonnet-4-5",
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
    }
    if ts is not None:
        record["timestamp"] = ts
    return record


def test_an_undated_transcript_record_is_skipped_and_counted(tmp_path):
    import json

    from tokenjam.core.backfill import parse_claude_code_session

    path = tmp_path / "sess-1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        _assistant_record("a", "2026-07-01T10:00:00.000Z"),
        _assistant_record("b", None),
        _assistant_record("c", "not-a-timestamp"),
    ]))

    parsed = parse_claude_code_session(path)

    assert parsed is not None
    assert parsed.records_undated == 2
    assert len(parsed.spans) == 1
    assert parsed.started_at == datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)


def test_a_transcript_with_no_dated_record_at_all_yields_no_session(tmp_path):
    """Returning one would invent both its start and its extent."""
    import json

    from tokenjam.core.backfill import parse_claude_code_session

    path = tmp_path / "sess-2.jsonl"
    path.write_text(json.dumps(_assistant_record("a", None, session_id="sess-2")))

    assert parse_claude_code_session(path) is None


# --- the one-shot cleanup for corpora an older build already wrote ----------


@pytest.fixture
def store(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "telemetry.duckdb")))
    yield db
    db.close()


def _write_sentinel_rows(store) -> None:
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    store.conn.execute(
        "INSERT INTO spans (span_id, trace_id, name, kind, status_code, start_time)"
        " VALUES ('sentinel','t','x','internal','ok',?)", [epoch],
    )
    store.conn.execute(
        "INSERT INTO spans (span_id, trace_id, name, kind, status_code, start_time)"
        " VALUES ('real','t','x','internal','ok',?)", [now],
    )
    store.conn.execute(
        "INSERT INTO sessions (session_id, agent_id, started_at)"
        " VALUES ('sentinel','a',?)", [epoch],
    )
    store.conn.execute(
        "INSERT INTO sessions (session_id, agent_id, started_at)"
        " VALUES ('real','a',?)", [now],
    )


def test_sentinel_rows_are_counted_per_table(store):
    _write_sentinel_rows(store)
    assert count_sentinel_timestamp_rows(store.conn) == {"spans": 1, "sessions": 1}


def test_a_clean_store_reports_nothing(store):
    store.conn.execute(
        "INSERT INTO spans (span_id, trace_id, name, kind, status_code, start_time)"
        " VALUES ('real','t','x','internal','ok',?)",
        [datetime.now(tz=timezone.utc)],
    )
    assert count_sentinel_timestamp_rows(store.conn) == {}


def test_the_purge_deletes_them_and_reports_how_many(store):
    """Deleted, not corrected: there is nothing to correct TO. A row whose only
    recorded fact about time was false has no other evidence of when it
    happened, and keeping it would leave a row every COUNT(*) counts and nothing
    can place on a calendar."""
    _write_sentinel_rows(store)

    removed = purge_sentinel_timestamp_rows(store.conn)

    assert removed == {"spans": 1, "sessions": 1}
    assert [r[0] for r in store.conn.execute(
        "SELECT span_id FROM spans").fetchall()] == ["real"]
    assert [r[0] for r in store.conn.execute(
        "SELECT session_id FROM sessions").fetchall()] == ["real"]
    assert count_sentinel_timestamp_rows(store.conn) == {}


def test_the_purge_is_idempotent(store):
    _write_sentinel_rows(store)
    purge_sentinel_timestamp_rows(store.conn)
    assert purge_sentinel_timestamp_rows(store.conn) == {}


def test_doctor_offers_the_cleanup_and_repair_takes_it(store):
    """Offered through `tj doctor`, not as a SQL snippet somebody has to be told
    about. A warning rather than an error: the rows are inert right up until
    something takes a naive MIN() over them."""
    from tokenjam.cli.cmd_doctor import _attempt_repairs, _check_sentinel_timestamps

    assert _check_sentinel_timestamps(store)["level"] == "ok"

    _write_sentinel_rows(store)
    check = _check_sentinel_timestamps(store)
    assert check["level"] == "warning"
    assert check["repair_action"] == "purge_timestamp_sentinels"
    assert "1 in sessions" in check["message"] and "1 in spans" in check["message"]

    _attempt_repairs([check], store, output_json=True)

    assert _check_sentinel_timestamps(store)["level"] == "ok"
    assert count_sentinel_timestamp_rows(store.conn) == {}


def test_a_session_start_can_be_corrected_by_a_genuinely_earlier_span(store):
    """`started_at` was absent from upsert_session's ON CONFLICT set list, so it
    was WRITE-ONCE: whatever the first span to reach a session stamped was
    permanent. Only `ended_at` ever advanced, so a session opened by an
    out-of-order span stayed wrong forever — which is why bad session timestamps
    accumulated in a corpus instead of healing.
    """
    from tests.factories import make_session

    late = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)
    early = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)

    store.upsert_session(make_session(session_id="s1", started_at=late, ended_at=late))
    store.upsert_session(make_session(session_id="s1", started_at=early, ended_at=early))

    stored = store.conn.execute(
        "SELECT started_at FROM sessions WHERE session_id = 's1'"
    ).fetchone()[0]
    assert stored == early


def test_a_later_span_never_pushes_a_session_start_forward(store):
    """A session starts when its EARLIEST observed span does, so the correction
    is one-directional — otherwise the last writer would win and the row would
    describe its most recent span rather than the session."""
    from tests.factories import make_session

    early = datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)
    late = datetime(2026, 7, 1, 18, 0, tzinfo=timezone.utc)

    store.upsert_session(make_session(session_id="s1", started_at=early, ended_at=early))
    store.upsert_session(make_session(session_id="s1", started_at=late, ended_at=late))

    stored = store.conn.execute(
        "SELECT started_at, ended_at FROM sessions WHERE session_id = 's1'"
    ).fetchone()
    assert stored[0] == early
    assert stored[1] == late
