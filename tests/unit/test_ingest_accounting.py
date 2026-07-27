"""Ingest-side accounting: one call counted once, one session summing to its spans.

Three defects that all end in a wrong dollar figure, pinned through the
PRODUCTION ingest paths rather than through the helpers they call — the
mechanism was already unit-tested when the money was still doubled, because
nothing wired it into a path a user reaches.

  1. **A call observed twice is priced twice.** A session that ran while
     `tj serve` was up is recorded live AND again when its transcript is
     backfilled; each path mints its own span_id, so span_id-keyed idempotency
     never sees the overlap.
  2. **A per-file session write that replaces instead of accumulating.** A
     Claude Code session is split across files sharing one session_id, so a
     replacing upsert leaves the row describing only the last file processed
     and `sessions.total_cost_usd` disagrees with `SUM(spans.cost_usd)`.
  3. **Capture reported as intent rather than as fact.** `capture_mode` echoed
     the config toggle, so a report could advertise prompt-prefix clustering
     while every cluster had clustered on tool signatures alone.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.api.routes.logs import parse_log_records
from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.config import CaptureConfig, StorageConfig, TjConfig
from tokenjam.core.cost import CostEngine
from tokenjam.core.db import (
    DuckDBBackend,
    InMemoryBackend,
    duplicate_call_observations,
    purge_duplicate_call_observations,
    session_cost_drift,
)
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.optimize import accounting
from tokenjam.core.optimize.types import ReuseFinding
from tokenjam.otel.semconv import ClaudeCodeEvents, GenAIAttributes, TjAttributes
from tests.factories import make_llm_span

SESSION_ID = "sess-accounting-1"
PROJECT = "-Users-dev-proj"
MODEL = "claude-sonnet-4-5-20250929"

#: Three assistant turns of one session, as (message id, prompt id, usage).
#: Token counts differ per turn, which is what a real conversation looks like:
#: the input grows as history accumulates.
_TURNS = (
    ("msg_aaa", "prompt-1", 12_000, 900, 4_000, 2_000),
    ("msg_bbb", "prompt-2", 19_000, 400, 6_500, 0),
    ("msg_ccc", "prompt-3", 25_000, 1_200, 1_000, 3_500),
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


# --- fixtures: the same three calls, as each ingest path sees them ------------

def _transcript_records(session_id: str, turns=_TURNS, cwd: str = "/Users/dev/proj"):
    records: list[dict] = []
    for i, (message_id, _prompt_id, inp, out, cache_r, cache_w) in enumerate(turns):
        records.append({
            "type": "user",
            "uuid": f"u-{message_id}",
            "sessionId": session_id,
            "cwd": cwd,
            "timestamp": (NOW + timedelta(seconds=i * 10)).isoformat(),
            "message": {"role": "user", "content": f"please do step {i}"},
        })
        records.append({
            "type": "assistant",
            "uuid": f"a-{message_id}",
            "sessionId": session_id,
            "cwd": cwd,
            "timestamp": (NOW + timedelta(seconds=i * 10 + 1)).isoformat(),
            "message": {
                "id": message_id,
                "model": MODEL,
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": inp,
                    "output_tokens": out,
                    "cache_read_input_tokens": cache_r,
                    "cache_creation_input_tokens": cache_w,
                },
            },
        })
    return records


def _write_transcript(root: Path, name: str, records: list[dict], sub: str = "") -> Path:
    directory = root / PROJECT / sub if sub else root / PROJECT
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _attr(key: str, value):
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _log_body(session_id: str, turns=_TURNS, *, with_prompts: bool = False) -> dict:
    """The same three calls as Claude Code's own OTel log exporter emits them.

    One `user_prompt` record per turn (carrying text only under
    OTEL_LOG_USER_PROMPTS=1, which `with_prompts` models) and one `api_request`
    record, which carries the token counts and no text at all.
    """
    records = []
    for i, (_message_id, prompt_id, inp, out, cache_r, cache_w) in enumerate(turns):
        ts = int((NOW + timedelta(seconds=i * 10)).timestamp() * 1e9)
        prompt_attrs = [
            _attr(ClaudeCodeEvents.SESSION_ID, session_id),
            _attr(ClaudeCodeEvents.PROMPT_ID, prompt_id),
        ]
        if with_prompts:
            prompt_attrs.append(_attr(ClaudeCodeEvents.PROMPT, f"please do step {i}"))
        records.append({
            "timeUnixNano": str(ts),
            "body": {"stringValue": ClaudeCodeEvents.USER_PROMPT},
            "attributes": prompt_attrs,
        })
        records.append({
            "timeUnixNano": str(ts + 1_000_000_000),
            "body": {"stringValue": ClaudeCodeEvents.API_REQUEST},
            "attributes": [
                _attr(ClaudeCodeEvents.SESSION_ID, session_id),
                _attr(ClaudeCodeEvents.PROMPT_ID, prompt_id),
                _attr(ClaudeCodeEvents.DURATION_MS, 1200),
                _attr("model", MODEL),
                _attr(ClaudeCodeEvents.INPUT_TOKENS, inp),
                _attr(ClaudeCodeEvents.OUTPUT_TOKENS, out),
                _attr(ClaudeCodeEvents.CACHE_READ_TOKENS, cache_r),
                _attr(ClaudeCodeEvents.CACHE_CREATION_TOKENS, cache_w),
            ],
        })
    return {"resourceLogs": [{
        "resource": {"attributes": [_attr("service.name", "claude-code")]},
        "scopeLogs": [{"logRecords": records}],
    }]}


@pytest.fixture
def db(tmp_path):
    backend = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    yield backend
    backend.close()


def _config(**capture_kwargs) -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(**capture_kwargs))


def _pipeline(backend, config: TjConfig | None = None) -> IngestPipeline:
    config = config or _config()
    return IngestPipeline(backend, config, cost_engine=CostEngine(backend))


def _live(backend, session_id=SESSION_ID, config=None, **kwargs) -> None:
    ingested, rejections = parse_log_records(
        _log_body(session_id, **kwargs), _pipeline(backend, config),
    )
    assert not rejections, rejections
    assert ingested


def _backfill(backend, root: Path, config: TjConfig | None = None):
    return ingest_claude_code(backend, root=root, config=config or _config())


def _totals(conn, session_id=SESSION_ID) -> tuple[int, float, int]:
    """(all-four-token-types, cost, llm span count) over a session's spans."""
    row = conn.execute(
        f"SELECT {accounting.four_type_token_sum_sql()}, "
        f"COALESCE(SUM(cost_usd), 0.0), COUNT(*) FROM spans "
        f"WHERE session_id = $1 AND model IS NOT NULL AND tool_name IS NULL",
        [session_id],
    ).fetchone()
    return int(row[0]), round(float(row[1]), 8), int(row[2])


# --- 1. one call, one count, whichever order the observers arrive in ----------

def test_backfill_after_live_ingest_does_not_double_count(db, tmp_path):
    _live(db)
    live_only = _totals(db.conn)
    assert live_only[2] == len(_TURNS)
    assert live_only[1] > 0.0        # the fixture actually costs something

    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))
    _backfill(db, tmp_path)

    assert _totals(db.conn) == live_only


def test_live_ingest_after_backfill_does_not_double_count(db, tmp_path):
    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))
    _backfill(db, tmp_path)
    backfill_only = _totals(db.conn)
    assert backfill_only[2] == len(_TURNS)

    pipeline = _pipeline(db)
    ingested, rejections = parse_log_records(_log_body(SESSION_ID), pipeline)
    assert not rejections, rejections

    assert _totals(db.conn) == backfill_only
    # Suppressed, not rejected: the batch is accepted in full and the calls it
    # restates simply are not written a second time.
    assert ingested == 2 * len(_TURNS)
    assert pipeline._duplicates_suppressed == len(_TURNS)


def test_both_ingest_orders_report_the_same_total(tmp_path):
    """The claim stated end to end: the reported figure cannot depend on which
    observer happened to see the session first."""
    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))

    orders: list[tuple[int, float, int]] = []
    for first_live in (True, False):
        backend = DuckDBBackend(StorageConfig(path=str(tmp_path / f"o{first_live}.duckdb")))
        try:
            if first_live:
                _live(backend)
                _backfill(backend, tmp_path)
            else:
                _backfill(backend, tmp_path)
                _live(backend)
            orders.append(_totals(backend.conn))
        finally:
            backend.close()

    live_first, backfill_first = orders
    assert live_first[0] == backfill_first[0]                 # tokens
    assert live_first[1] == pytest.approx(backfill_first[1])  # dollars
    assert live_first[2] == backfill_first[2] == len(_TURNS)  # calls


def test_a_repeated_backfill_stays_idempotent(db, tmp_path):
    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))
    _backfill(db, tmp_path)
    once = _totals(db.conn)
    _backfill(db, tmp_path)
    assert _totals(db.conn) == once


def test_a_call_only_one_observer_saw_is_still_counted(db, tmp_path):
    """Suppression is capped at what the other source actually recorded. A turn
    the live path missed is real work and must survive the backfill."""
    _live(db, turns=_TURNS[:2])
    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))
    _backfill(db, tmp_path)

    assert _totals(db.conn)[2] == len(_TURNS)


def test_two_identically_billed_calls_from_one_observer_are_two_calls(db):
    """A fingerprint may only collapse ACROSS ingest sources. One observer
    seeing the same billed shape twice saw two real calls, and dropping one
    would under-report spend."""
    pipeline = _pipeline(db)
    for i in range(2):
        pipeline.process(make_llm_span(
            session_id=SESSION_ID, agent_id="claude-code", model=MODEL,
            provider="anthropic", input_tokens=1_000, output_tokens=50,
            cache_tokens=0, cache_write_tokens=0,
            span_id=f"live-{i}", start_time=NOW + timedelta(seconds=i),
        ))
    assert _totals(db.conn)[2] == 2


def test_backfill_stamps_the_call_id_it_observed(db, tmp_path):
    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))
    _backfill(db, tmp_path)
    rows = db.conn.execute(
        "SELECT json_extract_string(attributes, '$.\"tj.call_id\"') FROM spans "
        "WHERE session_id = $1 AND model IS NOT NULL AND tool_name IS NULL",
        [SESSION_ID],
    ).fetchall()
    assert sorted(r[0] for r in rows) == sorted(t[0] for t in _TURNS)


def test_the_anthropic_patch_stamps_the_response_id():
    """The provider's own id names the call rather than this observation of
    it, which is what lets a second observer be recognised."""
    from tokenjam.sdk.integrations.anthropic import _record_response_id

    stamped: dict[str, object] = {}

    class _Span:
        def set_attribute(self, key, value):
            stamped[key] = value

    class _Response:
        id = "msg_01XYZ"

    _record_response_id(_Span(), _Response())
    assert stamped[GenAIAttributes.RESPONSE_ID] == "msg_01XYZ"

    stamped.clear()
    _record_response_id(_Span(), object())   # no id: unstamped, never raises
    assert stamped == {}


# --- 1b. legacy DBs: the doubled rows a prior build already wrote -------------

def _double_ingested_rows(backend, *, session_id=SESSION_ID) -> None:
    """Write the shape a pre-suppression build left behind: one live and one
    backfill-tagged observation per call, different span_ids, same billing."""
    for source in ("live", "backfill.claude_code"):
        for i, (message_id, _p, inp, out, cache_r, cache_w) in enumerate(_TURNS):
            extra = {} if source == "live" else {"source": source}
            backend.insert_span(make_llm_span(
                session_id=session_id, agent_id="claude-code", model=MODEL,
                provider="anthropic", input_tokens=inp, output_tokens=out,
                cache_tokens=cache_r, cache_write_tokens=cache_w,
                span_id=f"{source}-{message_id}",
                start_time=NOW + timedelta(seconds=i),
                cost_usd=1.0, extra_attributes=extra,
            ))


def test_a_legacy_double_ingested_db_is_detected_and_repaired(db):
    _double_ingested_rows(db)
    assert _totals(db.conn)[2] == 2 * len(_TURNS)

    spans, redundant_usd, worst = duplicate_call_observations(db.conn)
    assert spans == len(_TURNS)
    assert redundant_usd == pytest.approx(float(len(_TURNS)))
    assert worst and worst[0][0] == SESSION_ID

    deleted, sessions = purge_duplicate_call_observations(db.conn)
    assert deleted == len(_TURNS)
    assert sessions == [SESSION_ID]
    assert _totals(db.conn)[2] == len(_TURNS)


def test_the_repair_is_idempotent(db):
    _double_ingested_rows(db)
    purge_duplicate_call_observations(db.conn)
    assert duplicate_call_observations(db.conn)[0] == 0
    assert purge_duplicate_call_observations(db.conn) == (0, [])


def test_the_repair_leaves_a_singly_ingested_db_alone(db, tmp_path):
    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))
    _backfill(db, tmp_path)
    before = _totals(db.conn)
    assert purge_duplicate_call_observations(db.conn) == (0, [])
    assert _totals(db.conn) == before


def test_doctor_reports_and_repairs_a_double_ingested_db(db):
    """The user-facing route into the repair: a check that names the money and
    a `--repair` that leaves the sessions agreeing with their spans."""
    from tokenjam.cli.cmd_doctor import _attempt_repairs, _check_duplicate_call_ingest

    _double_ingested_rows(db)
    # A session row summing every stored span, i.e. the inflated figure a user
    # would have seen before the repair.
    db.recompute_session_totals_from_spans([SESSION_ID])

    check = _check_duplicate_call_ingest(db)
    assert check["level"] == "warning"
    assert check["repair_action"] == "drop_duplicate_calls"
    assert "$3.00" in check["message"]        # one redundant $1 span per turn

    _attempt_repairs([check], db, output_json=True)

    assert _totals(db.conn)[2] == len(_TURNS)
    assert _check_duplicate_call_ingest(db)["level"] == "ok"
    # The delete moved SUM(spans), so the repair has to reconcile the sessions
    # too or it just trades one disagreement for another.
    assert session_cost_drift(db.conn)[0] == 0


def test_doctor_is_quiet_on_a_singly_ingested_db(db, tmp_path):
    from tokenjam.cli.cmd_doctor import _check_duplicate_call_ingest

    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))
    _backfill(db, tmp_path)
    check = _check_duplicate_call_ingest(db)
    assert check["level"] == "ok"
    assert "repair_action" not in check


def test_duplicate_budget_never_suppresses_more_than_another_source_saw():
    assert accounting.duplicate_budget({"live": 2}, "backfill.claude_code") == 2
    assert accounting.duplicate_budget({"live": 2, "backfill.claude_code": 1},
                                       "backfill.claude_code") == 1
    # Already spent this run: the budget cannot be spent twice, which is what
    # keeps a genuinely repeated call from being dropped.
    assert accounting.duplicate_budget({"live": 1}, "backfill.claude_code", 1) == 0
    # More observations here than anywhere else is a gap THERE, never a
    # duplicate here.
    assert accounting.duplicate_budget({"live": 1, "backfill.claude_code": 3},
                                       "backfill.claude_code") == 0
    assert accounting.duplicate_budget({}, "live") == 0


# --- 2. a session row that adds up to its spans, with no repair pass ---------

def _multi_file_corpus(tmp_path: Path) -> Path:
    """A session as Claude Code stores one: a main-thread transcript plus a
    subagent file, both carrying the SAME internal sessionId."""
    _write_transcript(tmp_path, SESSION_ID, _transcript_records(SESSION_ID))
    _write_transcript(
        tmp_path, "agent-sub1",
        _transcript_records(SESSION_ID, turns=(
            ("msg_sub1", "prompt-s1", 5_000, 300, 100, 40),
            ("msg_sub2", "prompt-s2", 7_000, 250, 200, 60),
        )),
        sub=f"{SESSION_ID}/subagents",
    )
    return tmp_path


def _stored_session_totals(conn, session_id=SESSION_ID):
    row = conn.execute(
        "SELECT COALESCE(input_tokens,0), COALESCE(output_tokens,0), "
        "COALESCE(cache_tokens,0), COALESCE(cache_write_tokens,0), "
        "COALESCE(total_cost_usd,0.0) FROM sessions WHERE session_id = $1",
        [session_id],
    ).fetchone()
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), round(float(row[4]), 8))


def _span_sum_totals(conn, session_id=SESSION_ID):
    row = conn.execute(
        "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0), "
        "COALESCE(SUM(cache_tokens),0), COALESCE(SUM(cache_write_tokens),0), "
        "COALESCE(SUM(cost_usd),0.0) FROM spans WHERE session_id = $1",
        [session_id],
    ).fetchone()
    return (int(row[0]), int(row[1]), int(row[2]), int(row[3]), round(float(row[4]), 8))


def test_a_multi_file_backfill_adds_up_without_the_repair_pass(db, tmp_path, monkeypatch):
    """The per-file WRITE is what makes the row agree with its spans.

    The trailing `recompute_session_totals_from_spans` is disabled here on
    purpose: with it in place a replacing write and an accumulating one look
    identical, which is exactly why the defect survived. Prevention has to hold
    on its own — a windowed pass, a live ingest, or any future caller that
    never reaches the repair gets the same guarantee.
    """
    monkeypatch.delattr(DuckDBBackend, "recompute_session_totals_from_spans")
    _backfill(db, _multi_file_corpus(tmp_path))

    assert _stored_session_totals(db.conn) == _span_sum_totals(db.conn)
    assert session_cost_drift(db.conn)[0] == 0


def test_re_running_the_same_files_does_not_double_the_session(db, tmp_path, monkeypatch):
    monkeypatch.delattr(DuckDBBackend, "recompute_session_totals_from_spans")
    root = _multi_file_corpus(tmp_path)
    _backfill(db, root)
    once = _stored_session_totals(db.conn)

    _backfill(db, root)

    assert _stored_session_totals(db.conn) == once
    assert _stored_session_totals(db.conn) == _span_sum_totals(db.conn)


def test_the_session_row_carries_cache_writes(db, tmp_path, monkeypatch):
    """`session_record_from_parsed` never set `cache_write_tokens` at all, so
    the session row under-reported the priciest bucket until something else
    repaired it."""
    monkeypatch.delattr(DuckDBBackend, "recompute_session_totals_from_spans")
    _backfill(db, _multi_file_corpus(tmp_path))
    assert _stored_session_totals(db.conn)[3] > 0


def test_doctor_reports_no_cost_drift_on_a_freshly_backfilled_corpus(db, tmp_path):
    _backfill(db, _multi_file_corpus(tmp_path))
    count, total_drift, _worst = session_cost_drift(db.conn)
    assert (count, total_drift) == (0, 0.0)


def test_the_live_path_still_replaces_rather_than_accumulating(db):
    """The live path accumulates in Python before it writes, so its write must
    stay a replace — accumulating in SQL as well would double every figure."""
    pipeline = _pipeline(db)
    for i, (_m, _p, inp, out, cache_r, cache_w) in enumerate(_TURNS):
        pipeline.process(make_llm_span(
            session_id=SESSION_ID, agent_id="claude-code", model=MODEL,
            provider="anthropic", input_tokens=inp, output_tokens=out,
            cache_tokens=cache_r, cache_write_tokens=cache_w,
            span_id=f"live-{i}", start_time=NOW + timedelta(seconds=i),
        ))
    assert _stored_session_totals(db.conn) == _span_sum_totals(db.conn)


# --- 3. capture reported as measured, not as configured ----------------------

def test_the_live_path_carries_a_turns_prompt_onto_its_llm_span(db):
    """The backfill path attached prompt content and this one did not, so an
    analyzer wanting textual signal behaved differently depending only on how
    a session happened to be ingested."""
    _live(db, config=_config(prompts=True), with_prompts=True)
    rows = db.conn.execute(
        "SELECT json_extract_string(attributes, $1) FROM spans "
        "WHERE session_id = $2 AND model IS NOT NULL AND tool_name IS NULL",
        [f'$."{GenAIAttributes.PROMPT_CONTENT}"', SESSION_ID],
    ).fetchall()
    assert sorted(r[0] for r in rows) == [f"please do step {i}" for i in range(3)]


def test_capture_off_still_drops_the_live_prompt_content(db):
    """Attached at the converter, gated at the one ingest gate — so turning
    capture off keeps working exactly as it did."""
    _live(db, config=_config(prompts=False), with_prompts=True)
    rows = db.conn.execute(
        "SELECT json_extract_string(attributes, $1) FROM spans WHERE session_id = $2",
        [f'$."{GenAIAttributes.PROMPT_CONTENT}"', SESSION_ID],
    ).fetchall()
    assert all(r[0] is None for r in rows)


def test_an_exporter_that_sends_no_prompt_text_attaches_none(db):
    """Claude Code emits the turn text only under OTEL_LOG_USER_PROMPTS=1. The
    absence is honest here and is what the coverage measurement reports."""
    _live(db, config=_config(prompts=True), with_prompts=False)
    rows = db.conn.execute(
        "SELECT json_extract_string(attributes, $1) FROM spans WHERE session_id = $2",
        [f'$."{GenAIAttributes.PROMPT_CONTENT}"', SESSION_ID],
    ).fetchall()
    assert all(r[0] is None for r in rows)


def _reuse_finding(backend, config: TjConfig) -> ReuseFinding:
    from tokenjam.core.optimize import build_report

    report = build_report(
        backend, config,
        since=NOW - timedelta(days=1), until=NOW + timedelta(days=1),
        findings=["reuse"],
    )
    return report.findings["reuse"]


def _seed_planning_sessions(
    backend, *, with_prompt: bool, prompt_sessions: int | None = None,
) -> None:
    """Four sessions sharing one planning skeleton: an LLM call, then tools.

    `prompt_sessions` overrides the all-or-nothing `with_prompt` split so a
    window can be seeded with prompt text on only SOME planning calls — the
    mixed basis a real install produces when part of its history was ingested
    live without `OTEL_LOG_USER_PROMPTS=1`.
    """
    carrying = prompt_sessions if prompt_sessions is not None else (4 if with_prompt else 0)
    for s in range(4):
        session_id = f"plan-{s}"
        attrs = {"source": "backfill.claude_code"}
        if s < carrying:
            attrs[GenAIAttributes.PROMPT_CONTENT] = "cut a patch release"
        backend.insert_span(make_llm_span(
            session_id=session_id, agent_id="claude-code", model=MODEL,
            provider="anthropic", input_tokens=4_000, output_tokens=900,
            span_id=f"plan-llm-{s}", start_time=NOW,
            cost_usd=0.5, extra_attributes=attrs,
        ))
        for t, tool in enumerate(("Read", "Edit")):
            backend.insert_span(make_llm_span(
                session_id=session_id, agent_id="claude-code", model=None,
                tool_name=tool, span_id=f"plan-tool-{s}-{t}",
                start_time=NOW + timedelta(seconds=t + 1),
            ))


def test_capture_mode_reports_what_was_captured_not_what_was_configured():
    """The defect verbatim: capture on, no prompt text in the window, and the
    finding used to declare `with_prompt_prefix` while every cluster member's
    `prompt_prefix_hash` was None."""
    backend = InMemoryBackend()
    try:
        _seed_planning_sessions(backend, with_prompt=False)
        finding = _reuse_finding(backend, _config(prompts=True))
    finally:
        backend.close()

    assert finding.capture_mode == "tool_sequence_only"
    assert finding.prompt_capture_coverage == 0.0
    assert all(c.prompt_prefix_hash is None for c in finding.clusters)
    # The degrade is stated where the figure is explained, not only in a hint
    # a renderer may drop.
    assert "no planning call in the window carried prompt text" in finding.estimate_basis
    assert "[capture] prompts is on" in finding.hint


def test_capture_mode_reports_the_richer_mode_when_content_really_landed():
    backend = InMemoryBackend()
    try:
        _seed_planning_sessions(backend, with_prompt=True)
        finding = _reuse_finding(backend, _config(prompts=True))
    finally:
        backend.close()

    assert finding.capture_mode == "with_prompt_prefix"
    assert finding.prompt_capture_coverage == 1.0
    assert finding.clusters and all(
        c.prompt_prefix_hash is not None for c in finding.clusters
    )
    assert "every planning call in the window carried prompt text" in finding.estimate_basis


def test_coverage_is_none_when_there_was_nothing_to_measure():
    """`None` means "not measured", never 0.0 — the difference between an
    unanswered question and a measured absence."""
    backend = InMemoryBackend()
    try:
        finding = _reuse_finding(backend, _config(prompts=True))
    finally:
        backend.close()
    assert finding.prompt_capture_coverage is None


def test_partial_prompt_capture_is_reported_as_a_mixed_basis():
    """Nonzero coverage is not full coverage.

    Any prompt-bearing call used to flip the whole finding to
    `with_prompt_prefix`, while the basis string on the SAME finding said the
    rest clustered on tool sequence alone — the field and its own explanation
    contradicting each other. Every surface warns on `tool_sequence_only`
    only, so a half-guessed result rendered as the confident path.
    """
    backend = InMemoryBackend()
    try:
        _seed_planning_sessions(backend, with_prompt=False, prompt_sessions=2)
        finding = _reuse_finding(backend, _config(prompts=True))
    finally:
        backend.close()

    assert finding.capture_mode == "mixed_prompt_prefix"
    assert finding.prompt_capture_coverage == 0.5
    assert "2 of 4 planning calls carried prompt text" in finding.estimate_basis
    assert finding.hint and "Only some planning calls" in finding.hint


def test_every_degraded_capture_mode_is_flagged_as_degraded():
    """The renderers branch on this set, so it is what makes the notice fire.

    Pinned as a set rather than per-renderer: the partial case slipped through
    precisely because three separate surfaces each tested one literal value.
    """
    from tokenjam.core.optimize.types import DEGRADED_CAPTURE_MODES

    assert DEGRADED_CAPTURE_MODES == {"tool_sequence_only", "mixed_prompt_prefix"}
    assert "with_prompt_prefix" not in DEGRADED_CAPTURE_MODES
