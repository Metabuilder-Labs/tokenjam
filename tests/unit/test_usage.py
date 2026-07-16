"""Unit tests for core.usage — the shared Claude-Code usage parser.

This is the single source of truth both the statusline (cli/cmd_statusline) and
the backfill ingest path (core/backfill) read through, so the statusline's
re-read % cannot drift from the Cost tab. The tests lock the four-bucket parse,
the dedup KEY precedence, and the last-wins POLICY.
"""
from __future__ import annotations

import json

from tokenjam.core.usage import (
    AssistantUsage,
    assistant_message_key,
    iter_assistant_turns,
    iter_assistant_usage,
    iter_cumulative_usage,
    last_turn_context_tokens,
    parse_usage,
    session_usage,
)


def _line(**kwargs) -> str:
    msg_id = kwargs.pop("message_id", None)
    uuid = kwargs.pop("uuid", None)
    record = {"type": "assistant", "message": {"usage": kwargs}}
    if msg_id is not None:
        record["message"]["id"] = msg_id
    if uuid is not None:
        record["uuid"] = uuid
    return json.dumps(record)


# --- parse_usage ------------------------------------------------------------


def test_parse_usage_reads_four_buckets():
    usage = parse_usage({
        "input_tokens": 100, "output_tokens": 50,
        "cache_read_input_tokens": 800, "cache_creation_input_tokens": 50,
    })
    assert usage == AssistantUsage(100, 50, 800, 50)
    assert usage.total == 1000


def test_parse_usage_degrades_non_dict_and_missing_fields():
    assert parse_usage(None) == AssistantUsage(0, 0, 0, 0)
    assert parse_usage("garbage") == AssistantUsage(0, 0, 0, 0)
    assert parse_usage({"input_tokens": None}) == AssistantUsage(0, 0, 0, 0)


# --- assistant_message_key --------------------------------------------------


def test_message_key_precedence_id_then_uuid_then_line():
    assert assistant_message_key({"uuid": "u"}, {"id": "m"}, 7) == "m"
    assert assistant_message_key({"uuid": "u"}, {}, 7) == "u"
    assert assistant_message_key({}, {}, 7) == "7"


# --- iter_assistant_usage ---------------------------------------------------


def test_iter_skips_non_assistant_bad_and_empty_usage():
    lines = [
        json.dumps({"type": "user", "message": {"content": "hi"}}),
        "not json",
        _line(message_id="m0", input_tokens=0, output_tokens=0),  # empty usage
        _line(message_id="m1", input_tokens=500),
    ]
    got = list(iter_assistant_usage(lines))
    assert got == [("m1", AssistantUsage(500, 0, 0, 0))]


# --- session_usage (last-wins) ---------------------------------------------


def test_session_usage_last_wins_on_growing_snapshots():
    lines = [
        _line(message_id="m1", input_tokens=100, output_tokens=10),
        _line(message_id="m1", input_tokens=100, output_tokens=400),  # final
    ]
    assert session_usage(lines) == AssistantUsage(100, 400, 0, 0)


def test_session_usage_sums_distinct_messages():
    lines = [
        _line(message_id="m1", input_tokens=100),
        _line(message_id="m2", cache_read_input_tokens=300),
    ]
    assert session_usage(lines) == AssistantUsage(100, 0, 300, 0)


def test_session_usage_no_id_records_counted_separately():
    # Without a message id, each record falls back to a distinct key (uuid/line)
    # so genuinely separate turns are not collapsed.
    lines = [
        _line(uuid="a", input_tokens=100),
        _line(uuid="b", input_tokens=100),
    ]
    assert session_usage(lines).total == 200


# --- last_turn_context_tokens (live window occupancy) -----------------------


def test_last_turn_context_tokens_uses_final_turn_non_output():
    # Window occupancy = the last turn's input + cache_read + cache_write
    # (output isn't part of the prompt that was sent). The last DISTINCT turn.
    lines = [
        _line(message_id="m1", input_tokens=100, cache_read_input_tokens=50),
        _line(message_id="m2", input_tokens=10, cache_read_input_tokens=180_000,
              cache_creation_input_tokens=2_000, output_tokens=9_999),
    ]
    assert last_turn_context_tokens(lines) == 10 + 180_000 + 2_000


def test_last_turn_context_tokens_last_wins_on_growing_snapshots():
    lines = [
        _line(message_id="m1", input_tokens=100, cache_read_input_tokens=10),
        _line(message_id="m1", input_tokens=100, cache_read_input_tokens=500),  # final
    ]
    assert last_turn_context_tokens(lines) == 600


def test_last_turn_context_tokens_zero_when_no_turns():
    assert last_turn_context_tokens(["not json", '{"type": "user"}']) == 0


# --- iter_assistant_turns (model + usage per record) ------------------------


def _line_with_model(model: str, **kwargs) -> str:
    line = json.loads(_line(**kwargs))
    line["message"]["model"] = model
    return json.dumps(line)


def test_iter_assistant_turns_surfaces_model_alongside_usage():
    lines = [
        _line_with_model("claude-opus-4-8", message_id="m1", input_tokens=100),
        _line_with_model("claude-sonnet-4-5", message_id="m2", input_tokens=200),
    ]
    got = [(key, model) for key, model, _usage in iter_assistant_turns(lines)]
    assert got == [("m1", "claude-opus-4-8"), ("m2", "claude-sonnet-4-5")]


def test_iter_assistant_turns_same_filter_as_iter_assistant_usage():
    lines = [
        json.dumps({"type": "user", "message": {"content": "hi"}}),
        "not json",
        _line(message_id="m0", input_tokens=0, output_tokens=0),  # empty usage
        _line_with_model("claude-opus-4-8", message_id="m1", input_tokens=500),
    ]
    got = list(iter_assistant_turns(lines))
    assert got == [("m1", "claude-opus-4-8", AssistantUsage(500, 0, 0, 0))]


# --- iter_cumulative_usage (point-in-time preview walk) ----------------------


def test_iter_cumulative_usage_accumulates_across_distinct_turns():
    lines = [
        _line_with_model("claude-opus-4-8", message_id="m1", input_tokens=100),
        _line_with_model("claude-opus-4-8", message_id="m2", cache_read_input_tokens=300),
    ]
    got = list(iter_cumulative_usage(lines))
    assert got == [
        (1, "claude-opus-4-8", AssistantUsage(100, 0, 0, 0)),
        (2, "claude-opus-4-8", AssistantUsage(100, 0, 300, 0)),
    ]


def test_iter_cumulative_usage_growing_snapshot_does_not_advance_turn():
    # A mid-stream growing snapshot under the SAME message id updates the
    # running total in place — it is not a new turn.
    lines = [
        _line_with_model("claude-opus-4-8", message_id="m1", input_tokens=100, output_tokens=10),
        _line_with_model("claude-opus-4-8", message_id="m1", input_tokens=100, output_tokens=400),
    ]
    got = list(iter_cumulative_usage(lines))
    assert [turn for turn, _model, _usage in got] == [1, 1]
    assert got[-1][2] == AssistantUsage(100, 400, 0, 0)


def test_iter_cumulative_usage_final_total_matches_session_usage():
    lines = [
        _line_with_model("claude-opus-4-8", message_id="m1", input_tokens=100, output_tokens=10),
        _line_with_model("claude-opus-4-8", message_id="m1", input_tokens=100, output_tokens=400),
        _line_with_model("claude-sonnet-4-5", message_id="m2", cache_read_input_tokens=5000),
    ]
    *_rest, last = iter_cumulative_usage(lines)
    _turn, _model, final = last
    assert final == session_usage(lines)
