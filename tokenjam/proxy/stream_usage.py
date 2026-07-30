"""Streaming usage-payload observation for the pass-through proxy.

A streamed provider response reports its token usage only in a FINAL payload.
On Anthropic that is the trailing ``message_delta``; on OpenAI-compatible APIs
it is an extra trailing chunk that is emitted **only** when the request carried
``stream_options={"include_usage": true}``. If the caller never opted in, or
the connection drops before the final payload arrives, the call is recorded
with no token counts — which is indistinguishable from a call that cost
nothing, so the spend total silently reads LOW.

This module gives the proxy a way to notice that, without changing what it
forwards. :class:`SseUsageScanner` watches the bytes as they pass through and
answers three questions — was this a stream, did a usage payload arrive, was
any content produced — and :func:`stream_usage_span` records the answers as a
span the ``stream-usage`` analyzer reads later.

Pass-through is sacred (see ``proxy/app.py``): every function here swallows its
own errors. A malformed chunk makes the scanner less certain, never breaks the
relay.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from tokenjam.core.models import NormalizedSpan, SpanKind, SpanStatus
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.utils.ids import new_span_id, new_trace_id
from tokenjam.utils.time_parse import utcnow

logger = logging.getLogger("tokenjam.proxy")

# The span the proxy emits per observed stream. Distinct from the policy
# self-observation span so the two never share a row: this one is about data
# quality, that one is about enforcement.
STREAM_USAGE_SPAN_NAME = "tokenjam.stream.usage"


@dataclass(frozen=True)
class StreamUsageObservation:
    """What one streamed response through the proxy turned out to be."""
    provider:        str
    model:           str | None
    usage_reported:  bool
    content_chunks:  int
    # OpenAI-compatible only: did the request opt in to a usage chunk at all?
    # None for providers where the flag does not exist (Anthropic always
    # reports usage on a completed stream), which keeps "the caller did not
    # ask" distinguishable from "the caller asked and it never arrived" —
    # they have different remediations.
    usage_opt_in:    bool | None = None

    @property
    def usage_missing(self) -> bool:
        """The signature: content was produced and no usage payload arrived."""
        return self.content_chunks > 0 and not self.usage_reported


def stream_requested(body: dict | None) -> bool:
    """True when the request body asked for a streamed response."""
    return bool((body or {}).get("stream"))


def usage_opt_in(provider: str, body: dict | None) -> bool | None:
    """Whether an OpenAI-compatible request opted in to the usage chunk.

    None for providers that have no such flag — see
    :attr:`StreamUsageObservation.usage_opt_in`.
    """
    if provider == "anthropic":
        return None
    options = (body or {}).get("stream_options")
    if not isinstance(options, dict):
        return False
    return bool(options.get("include_usage"))


def requested_model(body: dict | None) -> str | None:
    model = (body or {}).get("model")
    return model if isinstance(model, str) and model else None


def _event_reports_usage(provider: str, event: dict) -> bool:
    """True when this SSE event is the one carrying final token counts."""
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return False
    if provider == "anthropic":
        # `message_start` also carries a `usage` block, but its `output_tokens`
        # is a placeholder for a response that has not been generated yet. Only
        # the trailing `message_delta` reports the real total, so accepting any
        # usage block here would mark every interrupted stream as complete.
        return event.get("type") == "message_delta"
    return any(k in usage for k in ("completion_tokens", "total_tokens", "output_tokens"))


def _event_carries_content(event: dict) -> bool:
    """True when this SSE event delivered generated output to the caller."""
    delta = event.get("delta")
    if isinstance(delta, dict) and (delta.get("text") or delta.get("partial_json")):
        return True
    for choice in event.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        choice_delta = choice.get("delta")
        if isinstance(choice_delta, dict) and (
            choice_delta.get("content") or choice_delta.get("tool_calls")
        ):
            return True
    return False


class SseUsageScanner:
    """Accumulates SSE bytes and answers whether a usage payload arrived.

    Line-buffered, because a provider chunk boundary can fall anywhere — a
    naive per-chunk parse would miss a usage payload split across two reads and
    report a healthy stream as broken.
    """

    def __init__(self, provider: str, model: str | None = None,
                 opt_in: bool | None = None) -> None:
        self._provider = provider
        self._model = model
        self._opt_in = opt_in
        self._buffer = b""
        self._usage_reported = False
        self._content_chunks = 0

    def feed(self, chunk: bytes) -> None:
        """Scan one relayed chunk. Never raises — the relay comes first."""
        try:
            self._feed(chunk)
        except Exception:  # scanning must never break pass-through
            logger.debug("proxy stream-usage scan failed (ignored)", exc_info=True)

    def _feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            self._scan_line(line)

    def _scan_line(self, line: bytes) -> None:
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return
        payload = stripped[len(b"data:"):].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            event = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        if _event_reports_usage(self._provider, event):
            self._usage_reported = True
        if _event_carries_content(event):
            self._content_chunks += 1

    def result(self) -> StreamUsageObservation:
        return StreamUsageObservation(
            provider=self._provider,
            model=self._model,
            usage_reported=self._usage_reported,
            content_chunks=self._content_chunks,
            usage_opt_in=self._opt_in,
        )


async def tap_stream(
    source: AsyncIterator[bytes],
    scanner: SseUsageScanner,
    sink: Callable[[StreamUsageObservation], None] | None,
) -> AsyncIterator[bytes]:
    """Relay ``source`` verbatim while ``scanner`` watches it go past.

    The ``finally`` is the whole point: it runs when the response completes AND
    when the client disconnects mid-stream (the generator is closed), so the
    disconnect case — the one that under-counts — is the case that is recorded.
    """
    try:
        async for chunk in source:
            scanner.feed(chunk)
            yield chunk
    finally:
        if sink is not None:
            try:
                sink(scanner.result())
            except Exception:  # recording must never break pass-through
                logger.exception("proxy stream-usage recording failed (ignored)")


def stream_usage_span(obs: StreamUsageObservation, *, ts: Any = None) -> NormalizedSpan:
    """Build the data-quality span for one observed stream.

    Deliberately carries NO token counts: the whole point is that the real
    counts were never reported, and writing an estimate onto the span would
    feed a guess into the cost engine as though it were observed.
    """
    ts = ts or utcnow()
    attrs: dict[str, Any] = {
        TjAttributes.STREAMING: True,
        TjAttributes.STREAM_USAGE_REPORTED: obs.usage_reported,
        TjAttributes.STREAM_CONTENT_CHUNKS: obs.content_chunks,
        GenAIAttributes.PROVIDER_NAME: obs.provider,
    }
    if obs.model:
        attrs[GenAIAttributes.REQUEST_MODEL] = obs.model
    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=new_trace_id(),
        name=STREAM_USAGE_SPAN_NAME,
        kind=SpanKind.INTERNAL,
        status_code=SpanStatus.OK,
        start_time=ts,
        end_time=ts,
        duration_ms=0.0,
        attributes=attrs,
        provider=obs.provider,
        model=obs.model,
        billing_account=obs.provider,
    )


class StreamUsageSink:
    """Persists each observed stream as a span. Best-effort, like AuditSink."""

    def __init__(self, db: Any, pipeline: Any = None) -> None:
        self.db = db
        self.pipeline = pipeline

    def __call__(self, obs: StreamUsageObservation) -> None:
        try:
            span = stream_usage_span(obs)
            if self.pipeline is not None:
                self.pipeline.process(span)
            else:
                self.db.insert_span(span)
        except Exception:  # persistence must never break the proxy
            logger.exception("proxy stream-usage span emit failed (ignored)")
