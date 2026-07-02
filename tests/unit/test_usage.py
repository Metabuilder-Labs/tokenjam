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
    iter_assistant_usage,
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
