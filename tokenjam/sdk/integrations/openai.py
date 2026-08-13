"""
OpenAI provider integration.

Wraps openai.resources.chat.completions.Completions.create to automatically
create OTel spans with token usage and model attributes.

Also works for OpenAI-compatible providers (Groq, Together, Fireworks, xAI,
Azure OpenAI) — pass the provider's base_url and set provider name from it.
"""
from __future__ import annotations

import functools
import logging
from typing import Any

from opentelemetry import trace

from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.sdk.attribution import stamp_span_attribution
from tokenjam.sdk.integrations._request_capture import (
    extract_openai_completion,
    record_completion_content,
    record_full_request,
    record_prompt_content,
)

logger = logging.getLogger(__name__)


class OpenAIIntegration:
    name = "openai"
    installed = False

    def __init__(self, provider_name: str = "openai") -> None:
        self._original_create: Any = None
        self._tracer = None
        self._provider_name = provider_name

    def install(self, tracer) -> None:
        """Patch openai.resources.chat.completions.Completions.create."""
        if self.installed:
            return
        self._tracer = tracer
        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            logger.warning("openai package not installed — skipping patch")
            return

        self._original_create = Completions.create
        integration = self

        @functools.wraps(self._original_create)
        def patched_create(self_comp, *args, **kwargs):
            # Skip span if call originates from litellm (avoids double-counting)
            from tokenjam.sdk.integrations.litellm import _tj_litellm_active
            if _tj_litellm_active.get(False):
                return integration._original_create(self_comp, *args, **kwargs)
            span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, integration._provider_name)
            span.set_attribute(
                GenAIAttributes.REQUEST_MODEL,
                kwargs.get("model", "unknown"),
            )
            record_full_request(span, kwargs)
            # Prompt content (#320). Set unconditionally; stripped at ingest
            # unless [capture] prompts is on. Set before the call so it's present
            # even on the streaming path (completion text isn't aggregated there).
            record_prompt_content(span, kwargs.get("messages"))
            # Cost-attribution dimensions from the ambient sdk.attribution
            # context (#SDK dashboard shape) — this patched client call has no
            # per-call kwarg for tenant_id/feature.
            stamp_span_attribution(span)
            is_stream = kwargs.get("stream", False)
            try:
                response = integration._original_create(self_comp, *args, **kwargs)
                if is_stream:
                    return _StreamWrapper(response, span)
                if hasattr(response, "usage") and response.usage:
                    span.set_attribute(
                        GenAIAttributes.INPUT_TOKENS,
                        response.usage.prompt_tokens,
                    )
                    span.set_attribute(
                        GenAIAttributes.OUTPUT_TOKENS,
                        response.usage.completion_tokens,
                    )
                # Completion content (#320). Stripped at ingest unless
                # [capture] completions is on.
                record_completion_content(span, extract_openai_completion(response))
                span.set_status(trace.Status(trace.StatusCode.OK))
                span.end()
                return response
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                span.end()
                raise

        # setattr, not attribute assignment: `create` is an overloaded method on
        # a third-party class, and rebinding it directly is an error a type
        # checker is right to flag. The monkeypatch itself is the point here.
        setattr(Completions, "create", patched_create)
        self.installed = True
        logger.debug("OpenAI integration installed (provider=%s)", self._provider_name)

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            from openai.resources.chat.completions import Completions
            if self._original_create:
                setattr(Completions, "create", self._original_create)
        except ImportError:
            pass
        self.installed = False


def _chunk_carries_content(chunk: Any) -> bool:
    """True when a streamed chunk delivered generated output to the caller.

    Deliberately structural and defensive: the point is only to separate "this
    stream produced something the user was billed for" from "this stream was
    empty", so an unrecognised chunk shape counts as no content rather than
    raising into the caller's iteration.
    """
    choices = getattr(chunk, "choices", None)
    if not choices:
        return False
    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        if getattr(delta, "content", None) or getattr(delta, "tool_calls", None):
            return True
    return False


class _StreamWrapper:
    """Wraps an OpenAI stream to capture final usage chunk and end the span.

    Also records the streaming data-quality signature (see
    ``TjAttributes.STREAMING``): an OpenAI-compatible API emits usage ONLY when
    the request carried ``stream_options={"include_usage": true}`` AND the
    caller drains the iterator to the trailing chunk. Either omission leaves
    this span with no token counts, which is indistinguishable from a free call
    unless the span says so itself.
    """

    def __init__(self, stream, span):
        self._stream = stream
        self._span = span
        self._usage = None
        self._content_chunks = 0
        # Idempotency guard — see `_finalize`'s docstring.
        self._finalized = False

    def _consume(self, chunk: Any) -> None:
        if hasattr(chunk, "usage") and chunk.usage:
            self._usage = chunk.usage
        if _chunk_carries_content(chunk):
            self._content_chunks += 1

    def _finalize(self, ok: bool, exc: BaseException | None = None) -> None:
        """Record usage + the streaming data-quality signature, end the span.

        Reached from `__iter__`'s `finally` (drained/abandoned/errored via
        `for`) and from `__next__` on `StopIteration`/any exception (drained
        by hand via bare `next()`, which never touches `__iter__` at all —
        that used to leave the span never ended, hence never exported, no
        matter how the stream actually finished). Guarded so whichever path
        gets there first wins; a caller mixing both on one wrapper instance
        still only finalizes once.
        """
        if self._finalized:
            return
        self._finalized = True
        if self._usage:
            self._span.set_attribute(
                GenAIAttributes.INPUT_TOKENS,
                self._usage.prompt_tokens,
            )
            self._span.set_attribute(
                GenAIAttributes.OUTPUT_TOKENS,
                self._usage.completion_tokens,
            )
        # Stamped unconditionally, including on the happy path: the
        # analyzer needs the complete streams as the peer baseline it
        # estimates the missing ones against, so "usage reported" is as
        # load-bearing a fact as "usage missing".
        self._span.set_attribute(TjAttributes.STREAMING, True)
        self._span.set_attribute(
            TjAttributes.STREAM_USAGE_REPORTED, self._usage is not None,
        )
        self._span.set_attribute(
            TjAttributes.STREAM_CONTENT_CHUNKS, self._content_chunks,
        )
        if ok:
            self._span.set_status(trace.Status(trace.StatusCode.OK))
        else:
            self._span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc) if exc else ""))
        self._span.end()

    def __iter__(self):
        _ok = False
        _exc: BaseException | None = None
        try:
            for chunk in self._stream:
                self._consume(chunk)
                yield chunk
            _ok = True
        except Exception as exc:
            _exc = exc
            raise
        finally:
            self._finalize(ok=_ok, exc=_exc)

    def __next__(self):
        # Bare `next(wrapper)` used to bypass `__iter__` (and its `finally`)
        # entirely — `__iter__`'s body only runs under `for`/`list()`/etc, so
        # driving this by hand with `next()` in a loop left the span open
        # forever regardless of whether the stream finished cleanly.
        try:
            chunk = self._stream.__next__()
        except StopIteration:
            self._finalize(ok=True)
            raise
        except Exception as exc:
            self._finalize(ok=False, exc=exc)
            raise
        self._consume(chunk)
        return chunk


def patch_openai(base_url: str | None = None) -> None:
    """
    Wraps the OpenAI client.
    Also works for OpenAI-compatible providers (Groq, Together, Fireworks, xAI,
    Azure OpenAI) — pass the provider's base_url and set provider name from it.
    """
    from tokenjam.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    provider = "openai"
    if base_url:
        # Infer provider name from base_url domain
        from urllib.parse import urlparse
        domain = urlparse(base_url).hostname or ""
        if "groq" in domain:
            provider = "groq"
        elif "together" in domain:
            provider = "together"
        elif "fireworks" in domain:
            provider = "fireworks"
        elif "xai" in domain or "x.ai" in domain:
            provider = "xai"
        elif "azure" in domain:
            provider = "azure.openai"
        else:
            provider = domain.split(".")[0] if domain else "openai-compatible"

    integration = OpenAIIntegration(provider_name=provider)
    integration.install(trace.get_tracer("tokenjam.sdk"))
