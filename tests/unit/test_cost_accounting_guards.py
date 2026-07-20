"""Accounting guards for the cost surfaces.

Two failure modes that both quietly corrupt a dollar figure:

  1. The same LLM call reaching the store twice (one ingest path observes it
     live, another backfills it from the transcript), so a row-level sum prices
     it twice. Every figure a cost card prints has to be summed over CALLS.
  2. An aggregate that adds up token types and forgets ``cache_write_tokens``,
     which under-reports the most expensive bucket.

Scope note, so these tests are read for what they prove: the fixture spans
below stamp each observation with the identity of the call it describes. That
is what the accounting layer keys on, and it is the contract the ingest paths
have to meet; making them agree on that stamp is separate work on its own
track. These tests pin the CONSUMING side: given identically-identified
observations, a cost figure is summed per call, in either arrival order, and a
span carrying no identity still counts as its own call.

Fully isolated: an ``InMemoryBackend`` DB, no config or transcript reads.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import accounting, cost_verify
from tokenjam.core.optimize import runner as runner_mod
from tokenjam.core.optimize.analyzers import (
    budget_projection,
    cache_efficacy,
    model_downgrade,
    output_verbosity,
    prompt_bloat,
    subagent_rightsizing,
    workflow_restructure,
)
from tests.factories import make_llm_span
from tests.token_aggregate_guard import (
    TOKEN_COLUMNS,
    assert_module_token_sums_are_complete,
    assert_sql_token_sums_are_complete,
    find_incomplete_token_sums,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
WINDOW_START = NOW - timedelta(days=2)

#: One fixture session: three assistant turns, each a distinct API call.
_CALLS = (
    ("msg_aaa", 12_000, 900, 4_000, 2_000),
    ("msg_bbb", 9_000, 400, 6_500, 0),
    ("msg_ccc", 15_000, 1_200, 1_000, 3_500),
)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _span(call, *, source: str, offset_ms: int):
    """One observation of one call.

    ``source`` distinguishes the two ingest paths: they mint different
    ``span_id``s for the same call and land milliseconds apart, which is
    exactly why span_id-keyed dedup does not catch the overlap. Both stamp the
    call's identity, which is what the accounting layer keys on.
    """
    call_id, input_tokens, output_tokens, cache_read, cache_write = call
    return make_llm_span(
        agent_id="svc-a", session_id="sess-1", model="claude-sonnet-5",
        provider="anthropic",
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_tokens=cache_read, cache_write_tokens=cache_write,
        span_id=f"{source}-{call_id}",
        start_time=WINDOW_START + timedelta(milliseconds=offset_ms),
        extra_attributes={"tj.call_id": call_id, "source": source},
    )


def _live_hook_spans():
    return [_span(c, source="live", offset_ms=i * 1000) for i, c in enumerate(_CALLS)]


def _backfill_spans():
    # ~17ms after the live observation of the same call, per the shape observed
    # on real doubled sessions.
    return [_span(c, source="backfill.claude_code", offset_ms=i * 1000 + 17)
            for i, c in enumerate(_CALLS)]


def _priced_window(conn) -> float:
    """The dollars the cost surfaces would report for the fixture window."""
    rows = cost_verify._rows_for(conn, WINDOW_START - timedelta(days=1), NOW, "svc-a")
    return cost_verify._priced_usd(rows)


def _load(db, spans):
    for span in spans:
        db.insert_span(span)


# --- H1: call-identity-deduped accounting -------------------------------------

def test_ingest_order_does_not_change_the_dollar_figure(db):
    """Live hook first, then backfill."""
    _load(db, _live_hook_spans())
    live_then_backfill_before = _priced_window(db.conn)
    _load(db, _backfill_spans())
    live_then_backfill = _priced_window(db.conn)

    other = InMemoryBackend()
    try:
        _load(other, _backfill_spans())
        backfill_first = _priced_window(other.conn)
        _load(other, _live_hook_spans())
        backfill_then_live = _priced_window(other.conn)
    finally:
        other.close()

    # Both orders agree, and neither order changed the figure by observing the
    # same calls a second time.
    assert live_then_backfill == pytest.approx(backfill_then_live)
    assert live_then_backfill == pytest.approx(live_then_backfill_before)
    assert backfill_first == pytest.approx(backfill_then_live)


def test_a_double_ingested_session_is_priced_once(db):
    """The single-count claim, stated directly: a session observed by both
    ingest paths costs the same as the same session observed by one."""
    single = InMemoryBackend()
    try:
        _load(single, _live_hook_spans())
        expected = _priced_window(single.conn)
    finally:
        single.close()

    _load(db, _live_hook_spans())
    _load(db, _backfill_spans())

    assert _priced_window(db.conn) == pytest.approx(expected)
    assert expected > 0.0        # the fixture actually costs something


def test_distinct_calls_are_never_collapsed(db):
    """The dedup keys on call identity, not on the token counts: two real
    calls that happen to bill identically must both be counted."""
    twin = ("msg_ddd", 12_000, 900, 4_000, 2_000)   # same numbers as msg_aaa
    _load(db, _live_hook_spans())
    _load(db, [_span(twin, source="live", offset_ms=9_000)])

    single_only = InMemoryBackend()
    try:
        _load(single_only, _live_hook_spans())
        base = _priced_window(single_only.conn)
    finally:
        single_only.close()

    assert _priced_window(db.conn) > base


def test_spans_without_a_call_id_keep_their_row_identity(db):
    """No call id stamped means today's behaviour, unchanged: each row is its
    own call. The dedup must never silently merge them."""
    for i, call in enumerate(_CALLS):
        db.insert_span(make_llm_span(
            agent_id="svc-a", session_id="sess-1", model="claude-sonnet-5",
            input_tokens=call[1], output_tokens=call[2],
            cache_tokens=call[3], cache_write_tokens=call[4],
            start_time=WINDOW_START + timedelta(milliseconds=i * 1000),
        ))
    rows = cost_verify._rows_for(db.conn, WINDOW_START - timedelta(days=1), NOW, "svc-a")
    assert len(rows) == len(_CALLS)


def test_call_identity_prefers_the_stamped_id_over_the_span_id():
    stamped = accounting.call_identity(
        "span-1", "sess-1", {"tj.call_id": "msg_aaa"},
    )
    other_row_same_call = accounting.call_identity(
        "span-2", "sess-1", '{"tj.call_id": "msg_aaa"}',   # attributes as JSON text
    )
    assert stamped == other_row_same_call


def test_call_identity_falls_back_to_the_span_id():
    assert accounting.call_identity("span-1", "sess-1", None) != \
        accounting.call_identity("span-2", "sess-1", None)


def test_call_identity_separates_sessions():
    assert accounting.call_identity("span-1", "sess-1", {"tj.call_id": "m"}) != \
        accounting.call_identity("span-1", "sess-2", {"tj.call_id": "m"})


def test_dedupe_keeps_the_last_observation():
    rows = [("k1", 1), ("k2", 2), ("k1", 99)]
    assert accounting.dedupe_by_call_identity(rows) == [("k1", 99), ("k2", 2)]


# --- H2: every token aggregate sums all four types ----------------------------

#: Every module that builds a token aggregate. Add yours here when you add an
#: aggregate; that is the whole wiring cost of the guard.
_AGGREGATE_MODULES = [
    accounting, cost_verify, runner_mod,
    budget_projection, cache_efficacy, model_downgrade, output_verbosity,
    prompt_bloat, subagent_rightsizing, workflow_restructure,
]


@pytest.mark.parametrize(
    "module", _AGGREGATE_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1],
)
def test_module_token_aggregates_sum_all_four_token_types(module):
    assert_module_token_sums_are_complete(module)


def test_the_canonical_sum_names_every_token_type():
    sql = accounting.four_type_token_sum_sql(alias="tokens")
    for column in TOKEN_COLUMNS:
        assert column in sql
    assert_sql_token_sums_are_complete(sql, context="four_type_token_sum_sql")


def test_four_type_total_counts_both_cache_buckets():
    row = {"input_tokens": 1, "output_tokens": 2, "cache_tokens": 4, "cache_write_tokens": 8}
    assert accounting.four_type_token_total(row) == 15


def test_the_guard_catches_the_bug_shape_that_shipped_three_times():
    """The regression this whole helper exists for: a sum that looks complete
    and silently drops the cache-write bucket."""
    offenders = find_incomplete_token_sums(
        "SELECT SUM(input_tokens + output_tokens + cache_tokens) FROM spans"
    )
    assert offenders and offenders[0][1] == {"cache_write_tokens"}


def test_the_guard_allows_a_deliberate_single_bucket_sum():
    """A cache-read ratio legitimately sums one bucket on its own; the guard
    must not force a meaningless fourth term into it."""
    assert find_incomplete_token_sums(
        "SELECT COALESCE(SUM(cache_tokens), 0) AS cache_tok FROM spans"
    ) == []
