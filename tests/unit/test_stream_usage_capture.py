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

from tokenjam.otel.semconv import GenAIAttributes, TjAttributes


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


def test_openai_bare_next_consumption_still_survives_and_reports_tokens():
    """Before the fix, `__next__` bare-delegated straight to the underlying
    stream (`return self._stream.__next__()`), bypassing `__iter__`'s
    `finally` entirely — a caller driving the wrapper by hand with `next()`
    in a loop (never `for`/`list()`) left the span open forever, unexported,
    no matter how the stream actually finished. Drains fully via bare
    `next()` until `StopIteration` — the standard manual-iteration idiom —
    and asserts the span still ends with the trailing usage."""
    from tokenjam.sdk.integrations.openai import _StreamWrapper

    span = _FakeSpan()
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=40)
    chunks = [_openai_chunk(content="Hi"), _openai_chunk(content="there"),
              _openai_chunk(usage=usage)]
    wrapper = _StreamWrapper(iter(chunks), span)

    collected = []
    while True:
        try:
            collected.append(next(wrapper))
        except StopIteration:
            break

    assert len(collected) == 3
    assert span.ended
    assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 120
    assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 40
    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is True
    assert span.attributes[TjAttributes.STREAM_CONTENT_CHUNKS] == 2


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

    def __next__(self):
        # The real anthropic `MessageStream` supports direct `next()`
        # consumption (not just `for`), which is exactly what
        # `_StreamWrapper.__next__` bare-delegates to — needed to exercise
        # that path without going through `__iter__`'s generator at all.
        if self._consumed >= len(self._events):
            raise StopIteration
        event = self._events[self._consumed]
        self._consumed += 1
        return event

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


def test_anthropic_bare_next_consumption_still_survives_and_reports_tokens():
    """Before the fix, `.stream()`'s `_StreamWrapper.__next__` bare-delegated
    (`return next(self._stream)`), bypassing every finalize path — only
    `__exit__` (i.e. the `with` block) ever ended the span. `get_final_message`
    only needs the underlying stream fully drained, not `__exit__` called
    first, so draining by hand via bare `next()` until `StopIteration` — with
    no `with` block at all — must still report the real usage."""
    from tokenjam.sdk.integrations.anthropic import _StreamWrapper

    span = _FakeSpan()
    stream = _FakeAnthropicStream(
        [_anthropic_text_event("Hi"), _anthropic_text_event(" there")],
        final_usage=SimpleNamespace(input_tokens=90, output_tokens=12),
    )
    wrapper = _StreamWrapper(stream, span)

    collected = []
    while True:
        try:
            collected.append(next(wrapper))
        except StopIteration:
            break

    assert len(collected) == 2
    assert span.ended
    assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 90
    assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 12
    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is True


# ---------------------------------------------------------------------------
# Anthropic `create(stream=True)` — the raw SSE event stream, distinct from
# `.stream()`'s `MessageStream`. Before `_RawStreamWrapper` existed, this path
# fell through `patched_create`'s `hasattr(response, "usage")` check (a raw
# stream object never has one) and produced a ZERO-TOKEN span for every
# streamed `create()` call, however much was actually generated.
# ---------------------------------------------------------------------------

def _message_start_event(input_tokens, cache_read=None, cache_create=None, response_id="msg_01"):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_create,
    )
    message = SimpleNamespace(id=response_id, usage=usage)
    return SimpleNamespace(type="message_start", message=message, delta=None)


def _message_delta_event(output_tokens):
    usage = SimpleNamespace(output_tokens=output_tokens)
    return SimpleNamespace(type="message_delta", usage=usage, delta=None)


def test_anthropic_raw_create_stream_carries_real_tokens_via_for_loop():
    from tokenjam.sdk.integrations.anthropic import _RawStreamWrapper

    span = _FakeSpan()
    events = [
        _message_start_event(input_tokens=200, cache_read=50, cache_create=10),
        _anthropic_text_event("Hel"),
        _anthropic_text_event("lo"),
        _message_delta_event(output_tokens=42),
    ]
    wrapper = _RawStreamWrapper(iter(events), span)

    collected = list(wrapper)

    assert len(collected) == 4
    assert span.ended
    assert span.attributes[GenAIAttributes.RESPONSE_ID] == "msg_01"
    assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 200
    assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 42
    assert span.attributes[GenAIAttributes.CACHE_READ_TOKENS] == 50
    assert span.attributes[GenAIAttributes.CACHE_CREATE_TOKENS] == 10
    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is True
    assert span.attributes[TjAttributes.STREAM_CONTENT_CHUNKS] == 2


def test_anthropic_raw_create_stream_survives_bare_next_consumption():
    """The same linkage bare `create(stream=True)` needs: manual `next()`
    drives it (its context-manager protocol is optional, unlike `.stream()`'s
    `MessageStreamManager`), so this must ALSO finalize on exhaustion."""
    from tokenjam.sdk.integrations.anthropic import _RawStreamWrapper

    span = _FakeSpan()
    events = [
        _message_start_event(input_tokens=150),
        _message_delta_event(output_tokens=30),
    ]
    wrapper = _RawStreamWrapper(iter(events), span)

    collected = []
    while True:
        try:
            collected.append(next(wrapper))
        except StopIteration:
            break

    assert len(collected) == 2
    assert span.ended
    assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 150
    assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 30
    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is True


def test_anthropic_raw_create_stream_without_usage_events_reports_the_gap():
    """A stream that never emits message_start/message_delta (or is abandoned
    before them) must say so, not silently claim a free call."""
    from tokenjam.sdk.integrations.anthropic import _RawStreamWrapper

    span = _FakeSpan()
    wrapper = _RawStreamWrapper(iter([_anthropic_text_event("Hi")]), span)
    list(wrapper)

    assert span.attributes[TjAttributes.STREAM_USAGE_REPORTED] is False
    assert GenAIAttributes.INPUT_TOKENS not in span.attributes
    assert GenAIAttributes.OUTPUT_TOKENS not in span.attributes


# ---------------------------------------------------------------------------
# LiteLLM sync stream — same bare-`next()` gap as the OpenAI/Anthropic
# patches (litellm.py's `_SyncStreamWrapper.__next__` used to bare-delegate).
# Cache-token extraction (`_set_usage_attributes`/`_extract_cache_tokens`) is
# NOT touched here — it already worked correctly via `__iter__` before this
# fix and continues to, exercised by test_litellm_integration.py.
# ---------------------------------------------------------------------------

def test_litellm_sync_stream_survives_bare_next_consumption():
    from tokenjam.sdk.integrations.litellm import _SyncStreamWrapper

    span = _FakeSpan()
    usage = SimpleNamespace(
        prompt_tokens=80, completion_tokens=20,
        cache_read_input_tokens=None, prompt_tokens_details=None,
        cache_creation_input_tokens=None,
    )
    chunk1 = SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))])
    chunk2 = SimpleNamespace(usage=usage, choices=[])

    from tokenjam.sdk.integrations import litellm as litellm_mod
    real_token = litellm_mod._tj_litellm_active.set(True)

    wrapper = _SyncStreamWrapper(iter([chunk1, chunk2]), span, "claude-haiku-4-5", real_token)

    collected = []
    while True:
        try:
            collected.append(next(wrapper))
        except StopIteration:
            break

    assert len(collected) == 2
    assert span.ended
    assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 80
    assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 20
    # The contextvar token must be released exactly once — a double-reset
    # raises, which would otherwise crash the NEXT litellm call on this
    # thread rather than merely being redundant work.
    assert litellm_mod._tj_litellm_active.get(False) is False


def test_litellm_async_stream_survives_bare_anext_consumption():
    """Before this fix, `_AsyncStreamWrapper` had no `__anext__` at all — a
    manual `await wrapper.__anext__()` raised `AttributeError` immediately
    (Python's `async for` only ever drove `__aiter__()`'s separate generator,
    never this object's own `__anext__`). Drains by hand via real
    `await __anext__()` calls to a genuine `StopAsyncIteration` — the async
    parallel to the sync bare-`next()` idiom — and asserts the span
    finalizes exactly once with usage present."""
    import asyncio

    from tokenjam.sdk.integrations.litellm import _AsyncStreamWrapper

    async def _run():
        span = _FakeSpan()
        usage = SimpleNamespace(
            prompt_tokens=90, completion_tokens=30,
            cache_read_input_tokens=None, prompt_tokens_details=None,
            cache_creation_input_tokens=None,
        )
        chunk1 = SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))])
        chunk2 = SimpleNamespace(usage=usage, choices=[])

        async def _source():
            yield chunk1
            yield chunk2

        from tokenjam.sdk.integrations import litellm as litellm_mod
        real_token = litellm_mod._tj_litellm_active.set(True)

        wrapper = _AsyncStreamWrapper(_source(), span, "claude-haiku-4-5", real_token)

        collected = []
        while True:
            try:
                collected.append(await wrapper.__anext__())
            except StopAsyncIteration:
                break

        assert len(collected) == 2
        assert span.ended
        assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 90
        assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 30
        # Finalized exactly once — a second finalize would double-reset the
        # contextvar token, which raises (and would crash the next litellm
        # call on this task rather than merely repeat work).
        assert litellm_mod._tj_litellm_active.get(False) is False

    asyncio.run(_run())


def test_litellm_async_stream_finalizes_only_once_across_anext_and_aiter():
    """A caller could plausibly drain via bare __anext__() to exhaustion and
    THEN also exercise __aiter__ (unusual, but the guard must hold): the
    second path must be a genuine no-op, not a double-`span.end()` or a
    double contextvar reset."""
    import asyncio

    from tokenjam.sdk.integrations.litellm import _AsyncStreamWrapper

    async def _run():
        span = _FakeSpan()
        chunk = SimpleNamespace(usage=None, choices=[])

        async def _source():
            yield chunk

        from tokenjam.sdk.integrations import litellm as litellm_mod
        real_token = litellm_mod._tj_litellm_active.set(True)
        wrapper = _AsyncStreamWrapper(_source(), span, "claude-haiku-4-5", real_token)

        await wrapper.__anext__()
        with pytest.raises(StopAsyncIteration):
            await wrapper.__anext__()
        assert span.ended

        # __aiter__()'s generator is a SEPARATE object over the SAME
        # already-exhausted stream, so it immediately hits StopAsyncIteration
        # too — its own finally calls _finalize again, which must no-op.
        async for _ in wrapper:
            pass  # pragma: no cover - stream is exhausted, never yields

    asyncio.run(_run())


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
