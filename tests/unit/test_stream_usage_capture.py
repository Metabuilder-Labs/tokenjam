"""Recording the streaming usage gap at observation time.

A streamed response reports its token usage only in a final payload, and two
ordinary things stop that payload arriving: an OpenAI-compatible caller that
never sent ``stream_options={"include_usage": true}``, and a client that
disconnects before the stream finishes. Either way the call is recorded with no
token counts, which is indistinguishable from a free call.

These tests pin the three places that notice — the OpenAI provider patch, the
Anthropic provider patch, and the proxy's SSE tap — because the analyzer that
reports the gap can only ever be as good as what these stamp.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tokenjam.otel.semconv import TjAttributes


class _FakeSpan:
    """Records attributes so a wrapper can be tested without a TracerProvider."""

    def __init__(self) -> None:
        self.attributes: dict = {}
        self.ended = False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, _status):
        pass

    def end(self):
        self.ended = True


def _openai_chunk(content=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    choices = [SimpleNamespace(delta=delta)] if content is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


def _anthropic_text_event(text):
    return SimpleNamespace(delta=SimpleNamespace(text=text, partial_json=None))


# ---------------------------------------------------------------------------
# OpenAI-compatible provider patch
# ---------------------------------------------------------------------------

def test_openai_stream_without_include_usage_is_marked_as_missing_usage():
    # Arrange: a stream that produced content and never emitted a usage chunk,
    # which is exactly what an OpenAI-compatible API returns when the request
    # omitted stream_options={"include_usage": True}.
    from tokenjam.sdk.integrations.openai import _StreamWrapper

    span = _FakeSpan()
    chunks = [_openai_chunk(content="Hel"), _openai_chunk(content="lo")]

    # Act
    list(_StreamWrapper(iter(chunks), span))

    # Assert
    assert span.attributes[TjAttributes.STREAMING] is True
    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is False
    assert span.attributes[TjAttributes.STREAM_CONTENT_CHUNKS] == 2
    assert span.ended


def test_openai_stream_with_trailing_usage_chunk_is_marked_complete():
    from tokenjam.sdk.integrations.openai import _StreamWrapper

    span = _FakeSpan()
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=40)
    chunks = [_openai_chunk(content="Hi"), _openai_chunk(usage=usage)]

    list(_StreamWrapper(iter(chunks), span))

    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is True
    assert span.attributes[TjAttributes.STREAM_CONTENT_CHUNKS] == 1


def test_openai_stream_abandoned_midway_still_records_the_gap():
    """A client disconnect abandons the iterator; the span must still say so."""
    from tokenjam.sdk.integrations.openai import _StreamWrapper

    span = _FakeSpan()
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=40)
    chunks = [_openai_chunk(content="Hi"), _openai_chunk(content="there"),
              _openai_chunk(usage=usage)]

    iterator = iter(_StreamWrapper(iter(chunks), span))
    next(iterator)
    # The consumer goes away before the usage chunk arrives.
    iterator.close()

    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is False
    assert span.attributes[TjAttributes.STREAM_CONTENT_CHUNKS] == 1


# ---------------------------------------------------------------------------
# Anthropic provider patch
# ---------------------------------------------------------------------------

class _FakeAnthropicStream:
    """Anthropic's stream raises from get_final_message when not drained."""

    def __init__(self, events, final_usage=None, drained_required=True):
        self._events = list(events)
        self._final_usage = final_usage
        self._drained_required = drained_required
        self._consumed = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def __iter__(self):
        for event in self._events:
            self._consumed += 1
            yield event

    def get_final_message(self):
        if self._drained_required and self._consumed < len(self._events):
            raise RuntimeError("stream was not consumed to completion")
        return SimpleNamespace(usage=self._final_usage)


def test_anthropic_stream_interrupted_before_message_delta_is_marked_missing():
    # Arrange: two text deltas consumed, then the caller stops. Anthropic
    # reports output tokens only in the trailing message_delta, so nothing is
    # available — and get_final_message() raises rather than returning it.
    from tokenjam.sdk.integrations.anthropic import _StreamWrapper

    span = _FakeSpan()
    stream = _FakeAnthropicStream(
        [_anthropic_text_event("Hel"), _anthropic_text_event("lo"),
         _anthropic_text_event("!")],
        final_usage=SimpleNamespace(input_tokens=90, output_tokens=12),
    )

    # Act
    with _StreamWrapper(stream, span) as wrapped:
        iterator = iter(wrapped)
        next(iterator)
        next(iterator)

    # Assert
    assert span.attributes[TjAttributes.STREAMING] is True
    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is False
    assert span.attributes[TjAttributes.STREAM_CONTENT_CHUNKS] == 2
    assert span.ended


def test_anthropic_stream_drained_to_completion_is_marked_complete():
    from tokenjam.sdk.integrations.anthropic import _StreamWrapper

    span = _FakeSpan()
    stream = _FakeAnthropicStream(
        [_anthropic_text_event("Hi")],
        final_usage=SimpleNamespace(input_tokens=90, output_tokens=12),
    )

    with _StreamWrapper(stream, span) as wrapped:
        list(wrapped)

    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is True
    assert span.attributes[TjAttributes.STREAM_CONTENT_CHUNKS] == 1


def test_anthropic_get_final_message_raise_never_escapes_the_wrapper():
    """The raise is the ANSWER (stream not drained), never an error to leak."""
    from tokenjam.sdk.integrations.anthropic import _StreamWrapper

    span = _FakeSpan()

    class _Exploding(_FakeAnthropicStream):
        def get_final_message(self):
            raise RuntimeError("boom")

    stream = _Exploding([_anthropic_text_event("Hi")])
    with _StreamWrapper(stream, span) as wrapped:
        list(wrapped)

    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is False


# ---------------------------------------------------------------------------
# Streaming data-quality and call identity share one final message
# ---------------------------------------------------------------------------
#
# Both are read off `get_final_message()`, and the wrapper calls it exactly
# once. Nothing else in either feature's tests would notice if that single
# call started serving only one of them, so these two pin the pairing.

def test_anthropic_drained_stream_records_both_identity_and_usage():
    """A completed stream stamps the provider's response id AND its usage.

    The id names the API CALL rather than this observation of it, so a
    backfill that later restates the same call is recognised instead of
    counted twice. It is only available on the final message — the same
    object the usage comes from.
    """
    from tokenjam.otel.semconv import GenAIAttributes
    from tokenjam.sdk.integrations.anthropic import _StreamWrapper

    span = _FakeSpan()

    class _Identified(_FakeAnthropicStream):
        def get_final_message(self):
            message = super().get_final_message()
            message.id = "msg_stream_01"
            return message

    stream = _Identified(
        [_anthropic_text_event("Hi")],
        final_usage=SimpleNamespace(input_tokens=90, output_tokens=12),
    )

    with _StreamWrapper(stream, span) as wrapped:
        list(wrapped)

    assert span.attributes[GenAIAttributes.RESPONSE_ID] == "msg_stream_01"
    assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 12
    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is True


def test_anthropic_undrained_stream_stamps_no_identity_and_still_reports_the_gap():
    """No final message means neither fact is known — and saying so is the point.

    An unstamped span is what every span carried before call identity existed,
    so the accounting side degrades to its old behavior rather than inventing
    an id; the data-quality signature must still be recorded, since "usage
    missing" is the signal the analyzer exists to read.
    """
    from tokenjam.otel.semconv import GenAIAttributes
    from tokenjam.sdk.integrations.anthropic import _StreamWrapper

    span = _FakeSpan()
    stream = _FakeAnthropicStream(
        [_anthropic_text_event("Hel"), _anthropic_text_event("lo")],
        final_usage=SimpleNamespace(input_tokens=90, output_tokens=12),
    )

    with _StreamWrapper(stream, span) as wrapped:
        next(iter(wrapped))

    assert GenAIAttributes.RESPONSE_ID not in span.attributes
    assert span.attributes[TjAttributes.STREAMING] is True
    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is False


# ---------------------------------------------------------------------------
# Proxy SSE tap
# ---------------------------------------------------------------------------

def _sse(*events: dict) -> bytes:
    return b"".join(
        b"data: " + json.dumps(e).encode() + b"\n\n" for e in events
    )


def test_scanner_flags_openai_stream_that_never_emitted_usage():
    from tokenjam.proxy.stream_usage import SseUsageScanner

    scanner = SseUsageScanner("openai", model="gpt-4o-mini", opt_in=False)
    scanner.feed(_sse(
        {"choices": [{"delta": {"content": "Hel"}}], "usage": None},
        {"choices": [{"delta": {"content": "lo"}}], "usage": None},
    ))
    scanner.feed(b"data: [DONE]\n\n")

    result = scanner.result()
    assert result.usage_missing is True
    assert result.content_chunks == 2
    assert result.usage_opt_in is False


def test_scanner_accepts_openai_trailing_usage_chunk():
    from tokenjam.proxy.stream_usage import SseUsageScanner

    scanner = SseUsageScanner("openai", model="gpt-4o-mini", opt_in=True)
    scanner.feed(_sse(
        {"choices": [{"delta": {"content": "Hi"}}], "usage": None},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                                  "total_tokens": 13}},
    ))

    assert scanner.result().usage_missing is False


def test_scanner_ignores_anthropic_message_start_placeholder_usage():
    """message_start carries a placeholder output_tokens for an ungenerated
    response — treating it as the real total marks every interrupted stream
    complete, which is the exact failure this analyzer exists to catch."""
    from tokenjam.proxy.stream_usage import SseUsageScanner

    scanner = SseUsageScanner("anthropic", model="claude-haiku-4-5")
    scanner.feed(_sse(
        {"type": "message_start",
         "message": {"usage": {"input_tokens": 90, "output_tokens": 1}},
         "usage": {"input_tokens": 90, "output_tokens": 1}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
    ))

    result = scanner.result()
    assert result.usage_missing is True
    assert result.usage_opt_in is None  # Anthropic has no include_usage flag


def test_scanner_accepts_anthropic_message_delta_usage():
    from tokenjam.proxy.stream_usage import SseUsageScanner

    scanner = SseUsageScanner("anthropic", model="claude-haiku-4-5")
    scanner.feed(_sse(
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 12}},
    ))

    assert scanner.result().usage_missing is False


def test_scanner_survives_a_usage_payload_split_across_chunk_boundaries():
    """A provider chunk boundary can fall mid-JSON; a per-chunk parse would
    miss the usage payload and report a healthy stream as broken."""
    from tokenjam.proxy.stream_usage import SseUsageScanner

    raw = _sse(
        {"choices": [{"delta": {"content": "Hi"}}], "usage": None},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
    )
    scanner = SseUsageScanner("openai", model="gpt-4o-mini", opt_in=True)
    for i in range(0, len(raw), 7):
        scanner.feed(raw[i:i + 7])

    assert scanner.result().usage_missing is False


def test_scanner_never_raises_on_malformed_bytes():
    from tokenjam.proxy.stream_usage import SseUsageScanner

    scanner = SseUsageScanner("openai")
    scanner.feed(b"data: {not json at all\n\n")
    scanner.feed(b"\xff\xfe garbage \n")

    assert scanner.result().usage_missing is False  # nothing observed, no claim


@pytest.mark.asyncio
async def test_tap_records_on_client_disconnect_midstream():
    """The disconnect case is the one that under-counts, so it is the one the
    tap must report — the generator's `finally` is what makes that true."""
    from tokenjam.proxy.stream_usage import SseUsageScanner, tap_stream

    recorded = []

    async def _source():
        yield _sse({"choices": [{"delta": {"content": "Hi"}}], "usage": None})
        yield _sse({"choices": [], "usage": {"completion_tokens": 3}})

    scanner = SseUsageScanner("openai", model="gpt-4o-mini", opt_in=False)
    relay = tap_stream(_source(), scanner, recorded.append)
    assert await relay.__anext__()  # first chunk relayed
    await relay.aclose()            # client goes away

    assert len(recorded) == 1
    assert recorded[0].usage_missing is True


@pytest.mark.asyncio
async def test_tap_relays_bytes_unmodified():
    """Pass-through is sacred: observing must not alter a single byte."""
    from tokenjam.proxy.stream_usage import SseUsageScanner, tap_stream

    payload = [b"data: {\"choices\":[]}\n\n", b"data: [DONE]\n\n"]

    async def _source():
        for chunk in payload:
            yield chunk

    scanner = SseUsageScanner("openai")
    relayed = [chunk async for chunk in tap_stream(_source(), scanner, None)]

    assert relayed == payload
