"""Streaming calls that closed without reporting usage — a DATA-QUALITY finding.

A streamed response reports its token usage only in a final payload. On
Anthropic that is the trailing ``message_delta``; on OpenAI-compatible APIs it
is an extra trailing chunk emitted **only** when the request carried
``stream_options={"include_usage": true}``. Two ordinary things stop that
payload arriving: the caller never opted in, or the client disconnected before
the stream finished. Either way the call lands with no token counts, which
looks exactly like a call that cost nothing — so the spend total reads LOW and
nothing on any surface says so.

**This finding is not recoverable waste and must never be treated as any.** The
money was already spent and the provider already billed it; the fix makes the
figure CORRECT, it does not make it smaller. That is why this analyzer is
absent from ``cost_proposals.COST_ANALYZERS`` and why its finding carries no
``past_overspend_*`` field: those are the two mechanisms that would pull a
figure into the recoverable-waste rollup, and neither can reach it. See
Critical Rule 27 / 30 in ``tokenjam/CLAUDE.md`` for why mixing a
differently-derived figure into that total is a first-order bug.

**Persona.** This is an SDK-persona failure mode. An interactive coding agent
is not proxying end-user streaming chat and its harness drains its own streams,
so ``stream-usage`` is gated off for the ``claude-code`` persona in
``runner.PERSONA_DISABLED_ANALYZERS`` — it has no request-side lever there
either (the harness builds the request, per the Claude Code actionable
ceiling).

The signature is recorded at observation time, not inferred here: every path
that watches a stream — the SDK provider patches and the proxy's SSE tap —
stamps ``TjAttributes.STREAMING`` / ``STREAM_USAGE_REPORTED`` /
``STREAM_CONTENT_CHUNKS`` on the span. This module only reads them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from tokenjam.core.cost import calculate_cost
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.otel.semconv import TjAttributes

logger = logging.getLogger(__name__)

# The one sentence that keeps this figure out of the savings story wherever it
# is rendered. Carried on the finding as a dataclass default (the same device
# `MODEL_DOWNGRADE_CAVEAT` uses) so it cannot be dropped by a renderer that
# forgets it.
DATA_QUALITY_NOTE = (
    "This is a measurement gap, not recoverable waste. The money was already "
    "spent and the provider already billed it; capturing the usage payload "
    "makes the spend figure correct, it does not make it smaller. It is "
    "excluded from the recoverable-waste total for that reason."
)

# Providers whose streaming API needs an explicit request-side opt-in before a
# usage payload is emitted at all. Anthropic always reports usage on a stream
# that runs to completion, so its remediation is only about draining.
_OPENAI_COMPATIBLE_REMEDIATION = """\
stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True,
    # Without this the API emits NO usage payload for a streamed response.
    stream_options={"include_usage": True},
)
usage = None
for chunk in stream:
    # The usage chunk is the LAST one and carries an empty `choices` list, so
    # it only arrives if the iterator is drained. Consume the stream here,
    # server-side, rather than handing it to a client that may disconnect.
    if chunk.usage:
        usage = chunk.usage
"""

_ANTHROPIC_REMEDIATION = """\
with client.messages.stream(model=model, messages=messages) as stream:
    for event in stream:
        ...  # forward to your caller
    # Only reachable once the stream has been drained; a caller that
    # disconnects mid-response never gets here, so consume the stream
    # server-side and forward from what you have already accumulated.
    usage = stream.get_final_message().usage
"""


@dataclass
class StreamUsageCallSite:
    """One (agent, provider, model) whose streams lost their usage payload."""
    provider:             str
    model:                str | None
    agent_id:             str | None
    affected_calls:       int
    complete_calls:       int
    sessions:             int
    # Per-call peer medians the estimate is built from. None when this call
    # site has no completed stream to learn from — in which case both estimate
    # fields below are None too, never zero (Critical Rule 28's symmetric
    # degrade: a call site contributes to both sums or to neither).
    peer_input_tokens:    int | None = None
    peer_output_tokens:   int | None = None
    peer_cache_tokens:    int | None = None
    peer_cache_write_tokens: int | None = None
    undercounted_tokens:  int | None = None
    undercounted_usd:     float | None = None
    derivation:           str = ""
    remediation:          str = ""
    remediation_snippet:  str = ""


@dataclass
class StreamUsageFinding:
    """Streamed calls whose token counts were never reported."""
    streams_observed:      int = 0
    streams_missing_usage: int = 0
    call_sites:            list[StreamUsageCallSite] = field(default_factory=list)
    # Deliberately NOT named `past_overspend_*`. That name is the canonical
    # avoidable-spend field every cost analyzer publishes and the rollup sums;
    # this quantity is spend that WAS incurred and never recorded, a different
    # question over a different population. Reusing the name would put it in
    # the recoverable total by construction.
    undercounted_usd:      float | None = None
    undercounted_tokens:   int | None = None
    estimate_basis:        str = ""
    estimate_confidence:   str = "heuristic"
    accounting_note:       str = DATA_QUALITY_NOTE
    hint:                  str = ""


def _remediation_for(provider: str) -> tuple[str, str]:
    """(one-line instruction, copyable snippet) for a provider's stream API."""
    if provider == "anthropic":
        return (
            "Consume the stream to completion server-side and read "
            "`get_final_message().usage` — Anthropic reports output tokens only "
            "in the trailing `message_delta`, so an abandoned stream reports "
            "none at all.",
            _ANTHROPIC_REMEDIATION,
        )
    return (
        "Send `stream_options={\"include_usage\": True}` on the request AND "
        "drain the stream server-side — an OpenAI-compatible API emits no usage "
        "payload without the flag, and emits it only in the final chunk.",
        _OPENAI_COMPATIBLE_REMEDIATION,
    )


def _parse_attributes(raw: Any) -> dict | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _median_int(values: list[int]) -> int | None:
    return int(round(median(values))) if values else None


@dataclass
class _Bucket:
    """Mutable accumulator for one call site while scanning rows."""
    affected: int = 0
    sessions: set = field(default_factory=set)
    peer_input: list = field(default_factory=list)
    peer_output: list = field(default_factory=list)
    peer_cache: list = field(default_factory=list)
    peer_cache_write: list = field(default_factory=list)
    # Completed streams whose observer never saw token counts. Counted, not
    # discarded: "we watched N of these and learned nothing from them" is what
    # makes an unestimated gap legible instead of looking like no data at all.
    observed_without_tokens: int = 0

    @property
    def complete(self) -> int:
        """Completed streams the median can actually be taken over."""
        return len(self.peer_output)


def _select_streaming_spans(ctx: AnalyzerContext) -> list[tuple]:
    """Spans in the window that a stream observer stamped.

    Filtered in SQL on the streaming marker so this never scans the whole span
    table; the JSON path is quoted because the attribute keys contain dots.
    """
    clauses = [
        "start_time >= $1", "start_time < $2",
        f"json_extract_string(attributes, '$.\"{TjAttributes.STREAMING}\"') "
        f"IS NOT NULL",
    ]
    params: list[Any] = [ctx.since, ctx.until]
    if ctx.agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(ctx.agent_id)
    where = " AND ".join(clauses)
    # The token columns are read RAW, never COALESCEd to 0. NULL here means the
    # observer could not see token counts at all — the proxy's SSE tap watches
    # the wire and stamps the streaming signature without ever learning the
    # numbers — and that is a different fact from a call that reported zero.
    # Flattening the two lets token-less observations into the peer baseline
    # and drags the median toward zero; see `_bucket_rows`.
    return ctx.conn.execute(
        f"SELECT provider, model, agent_id, session_id, attributes, "
        f"input_tokens, output_tokens, cache_tokens, cache_write_tokens "
        f"FROM spans WHERE {where}",
        params,
    ).fetchall()


def _bucket_rows(rows: list[tuple]) -> tuple[dict, int, int]:
    """Fold spans into per-call-site buckets. Returns (buckets, seen, missing)."""
    buckets: dict[tuple, _Bucket] = {}
    seen = 0
    missing = 0
    for (provider, model, agent_id, session_id, attrs_raw,
         input_tokens, output_tokens, cache_tokens, cache_write_tokens) in rows:
        attrs = _parse_attributes(attrs_raw)
        if attrs is None or not attrs.get(TjAttributes.STREAMING):
            continue
        seen += 1
        key = (str(provider or "unknown"), model, agent_id)
        bucket = buckets.setdefault(key, _Bucket())
        if attrs.get(TjAttributes.STREAM_USAGE_REPORTED):
            # A completed stream is the peer baseline the missing ones are
            # estimated against, so it is recorded rather than skipped — but
            # only if it actually carries token counts. An observer can watch a
            # stream run to completion and still never learn its numbers (the
            # proxy tap reads the wire, not the provider's accounting), and
            # those spans store NULL. Admitting them as zeros would let the
            # median collapse toward zero and render a confident "$0.00" for a
            # gap nobody measured — the exact false all-clear this module
            # exists to prevent.
            if output_tokens is None and input_tokens is None:
                bucket.observed_without_tokens += 1
                continue
            bucket.peer_input.append(int(input_tokens or 0))
            bucket.peer_output.append(int(output_tokens or 0))
            bucket.peer_cache.append(int(cache_tokens or 0))
            bucket.peer_cache_write.append(int(cache_write_tokens or 0))
            continue
        # No usage payload. Only count it when content was actually produced:
        # a stream that opened and delivered nothing cost nothing to
        # under-count, and flagging it would inflate the gap with noise.
        if int(attrs.get(TjAttributes.STREAM_CONTENT_CHUNKS) or 0) <= 0:
            continue
        missing += 1
        bucket.affected += 1
        if session_id:
            bucket.sessions.add(session_id)
    return buckets, seen, missing


def _build_call_site(key: tuple, bucket: _Bucket, at) -> StreamUsageCallSite:
    provider, model, agent_id = key
    remediation, snippet = _remediation_for(provider)
    site = StreamUsageCallSite(
        provider=provider,
        model=model,
        agent_id=agent_id,
        affected_calls=bucket.affected,
        complete_calls=bucket.complete,
        sessions=len(bucket.sessions),
        remediation=remediation,
        remediation_snippet=snippet,
    )
    med_in = _median_int(bucket.peer_input)
    med_out = _median_int(bucket.peer_output)
    med_cache = _median_int(bucket.peer_cache)
    med_cache_write = _median_int(bucket.peer_cache_write)
    if med_out is None or not model:
        # Nothing observed to learn the per-call size from, or no model to
        # price it against. Both estimate fields stay None — an unmeasured
        # gap renders an em dash, never "$0.00", which would read as "no
        # blind spot" and is the exact lie this analyzer exists to prevent.
        if med_out is not None:
            site.derivation = (
                "The affected calls carry no model name, so there is no rate "
                "to price them at and no under-count figure is claimed."
            )
        elif bucket.observed_without_tokens:
            # Distinct from "nothing was observed": streams DID complete here,
            # the observer just never learned their token counts. Saying so
            # points at the actual remedy (instrument the SDK client, which
            # sees the usage payload) instead of implying an idle window.
            site.derivation = (
                f"{bucket.observed_without_tokens} completed stream"
                f"{'s' if bucket.observed_without_tokens != 1 else ''} "
                f"{'were' if bucket.observed_without_tokens != 1 else 'was'} "
                f"observed for this call site, but none carried token counts "
                f"— they were watched on the wire, where the provider's usage "
                f"payload is not accounted. There is no per-call baseline to "
                f"size the missing calls against, so no under-count figure is "
                f"claimed. Instrument the client itself (`patch_openai()` / "
                f"`patch_anthropic()`) to establish one."
            )
        else:
            site.derivation = (
                "No completed stream was observed for this call site in this "
                "window, so there is no per-call baseline to size the missing "
                "calls against and no under-count figure is claimed."
            )
        return site
    site.peer_input_tokens = med_in
    site.peer_output_tokens = med_out
    site.peer_cache_tokens = med_cache
    site.peer_cache_write_tokens = med_cache_write
    per_call_tokens = (
        (med_in or 0) + med_out + (med_cache or 0) + (med_cache_write or 0)
    )
    per_call_usd = calculate_cost(
        provider, model,
        input_tokens=med_in or 0,
        output_tokens=med_out,
        cache_read_tokens=med_cache or 0,
        cache_write_tokens=med_cache_write or 0,
        at=at,
    )
    site.undercounted_tokens = per_call_tokens * bucket.affected
    site.undercounted_usd = round(per_call_usd * bucket.affected, 6)
    site.derivation = (
        f"{bucket.affected} streamed call"
        f"{'s' if bucket.affected != 1 else ''} closed with no usage payload. "
        f"Each is sized at the MEDIAN of the {bucket.complete} completed stream"
        f"{'s' if bucket.complete != 1 else ''} for the same model in this "
        f"window ({med_in or 0} input + {med_out} output + {med_cache or 0} "
        f"cache-read + {med_cache_write or 0} cache-write tokens per call), "
        f"priced at that model's rates. Estimated, not observed: the real "
        f"counts for these calls were never reported by the provider."
        + (
            f" A further {bucket.observed_without_tokens} completed stream"
            f"{'s' if bucket.observed_without_tokens != 1 else ''} carried no "
            f"token counts and {'were' if bucket.observed_without_tokens != 1 else 'was'} "
            f"left out of that median rather than counted as zero."
            if bucket.observed_without_tokens else ""
        )
    )
    return site


@register("stream-usage")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a StreamUsageFinding to ctx.report.findings."""
    try:
        rows = _select_streaming_spans(ctx)
    except Exception:
        # An install whose spans predate the streaming attributes has nothing
        # to scan; a query failure must not take the whole report down.
        logger.debug("stream-usage span scan failed", exc_info=True)
        rows = []

    buckets, seen, missing = _bucket_rows(rows)

    finding = StreamUsageFinding(
        streams_observed=seen,
        streams_missing_usage=missing,
    )
    if seen == 0:
        finding.hint = (
            "No streamed calls were observed in this window. Streaming usage "
            "gaps are only visible on calls tokenjam watched as they streamed "
            "— instrument the provider client with `patch_openai()` / "
            "`patch_anthropic()`, or route traffic through `tj proxy`."
        )
        ctx.report.findings["stream-usage"] = finding
        return

    sites = [
        _build_call_site(key, bucket, ctx.until)
        for key, bucket in buckets.items()
        if bucket.affected > 0
    ]
    # Largest blind spot first; unpriced sites sort last rather than as zero.
    sites.sort(key=lambda s: (s.undercounted_usd is None, -(s.undercounted_usd or 0.0)))
    finding.call_sites = sites

    priced = [s for s in sites if s.undercounted_usd is not None]
    if priced:
        finding.undercounted_usd = round(sum(s.undercounted_usd or 0.0 for s in priced), 6)
        finding.undercounted_tokens = sum(s.undercounted_tokens or 0 for s in priced)
        unpriced = len(sites) - len(priced)
        finding.estimate_basis = (
            f"{missing} of {seen} observed streams closed without a usage "
            f"payload. Each is sized at the median per-call token usage of the "
            f"completed streams for the same model in this window and priced at "
            f"that model's rates"
            + (
                f"; {unpriced} call site{'s' if unpriced != 1 else ''} had no "
                f"completed stream to size against and contribute"
                f"{'' if unpriced != 1 else 's'} nothing to this figure."
                if unpriced else "."
            )
        )
    elif sites:
        tokenless = sum(b.observed_without_tokens for b in buckets.values())
        finding.estimate_basis = (
            f"{missing} of {seen} observed streams closed without a usage "
            f"payload, and no completed stream carrying token counts was "
            f"observed for any affected model in this window — so the size of "
            f"the gap is not estimated here, only its existence."
            + (
                f" {tokenless} stream{'s' if tokenless != 1 else ''} did "
                f"complete, but {'were' if tokenless != 1 else 'was'} watched "
                f"on the wire where the provider's usage payload is not "
                f"accounted, so {'they carry' if tokenless != 1 else 'it carries'} "
                f"no token counts to build a baseline from."
                if tokenless else ""
            )
        )

    ctx.report.findings["stream-usage"] = finding
