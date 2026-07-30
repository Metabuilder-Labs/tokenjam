"""Token and dollar accounting primitives shared by every cost surface.

Two rules live here, both of which have cost real money by being re-derived
per analyzer instead of shared.

**Four token types, always.** A span bills across four buckets: input, output,
cache READ (``cache_tokens``) and cache WRITE (``cache_write_tokens``). Cache
writes are the expensive ones (1.25x or 2x input), so an aggregate that omits
them under-reports exactly the traffic a cache card is about. That omission has
shipped three separate times in this package. ``four_type_token_sum_sql`` is
the canonical form; use it rather than hand-writing another ``SUM(...)``.

**Call identity, not row identity.** A single LLM call can reach the store more
than once: the live path and the backfill path both observe it, each minting
its own ``span_id``, so a row-level ``SUM`` counts it twice. Money figures must
therefore be summed over CALLS, not over rows. ``call_identity`` names the
call; ``dedupe_by_call_identity`` collapses repeats last-wins, matching the
policy ``core.usage.session_usage`` applies to transcript records (the last
record for a message carries the finalized usage).

Both ingest paths now stamp an explicit call id where one exists: the Anthropic
provider patch stamps ``gen_ai.response.id`` off the response, and the Claude
Code transcript backfill stamps ``tj.call_id`` with the assistant message key.
Without a stamp the fallback is the row's own ``span_id``, preserving the
store's historical behaviour exactly.

**The observation fingerprint, for observers that carry no id.** Claude Code's
own OTel exporter emits no response id at all — its ``api_request`` event
carries session, model, token counts and cost and nothing that names the call
(see ``ClaudeCodeEvents``). So the live-vs-backfill pair for a Claude Code
session cannot be matched on a stamped id from BOTH sides, however diligently
each side stamps. What both sides do carry is the call's billed shape:
``(session, model, input, output, cache read, cache write)``. That is
``call_fingerprint`` — derived from columns the store already has, so it reads
legacy rows written before any stamping existed.

A fingerprint is a weaker claim than an id, so it is only ever used to collapse
observations from DIFFERENT ingest sources (``ingest_source``). Two rows from
one observer are two real calls no matter how alike they bill; only a second
observer can restate a call the first already recorded. ``duplicate_budget``
turns that into the count a caller may suppress.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

#: The four billable token columns on a span row, in canonical order.
TOKEN_TYPE_COLUMNS: tuple[str, ...] = (
    "input_tokens", "output_tokens", "cache_tokens", "cache_write_tokens",
)

#: Attribute keys, in precedence order, that name the underlying API call.
#: First non-empty one wins. ``gen_ai.response.id`` is the provider's own id
#: for the response; ``tj.call_id`` is the internal stamp.
CALL_ID_ATTRIBUTE_KEYS: tuple[str, ...] = ("tj.call_id", "gen_ai.response.id")

#: Attribute naming the ingest path that recorded an observation.
INGEST_SOURCE_ATTRIBUTE = "source"

#: The ingest source of a span carrying no explicit one: the live receive path
#: (the SDK exporter, ``POST /api/v1/spans``, ``POST /api/v1/logs``).
LIVE_INGEST_SOURCE = "live"


def four_type_token_sum_sql(prefix: str = "", alias: str | None = None) -> str:
    """The canonical all-four-token-types SQL sum.

    ``prefix`` qualifies the columns (e.g. ``"s."``). Pass ``alias`` to append
    an ``AS <alias>``. Every new token aggregate should be built from this, so
    a missing cache bucket is impossible rather than merely discouraged.
    """
    inner = " + ".join(f"COALESCE({prefix}{col},0)" for col in TOKEN_TYPE_COLUMNS)
    sql = f"COALESCE(SUM({inner}), 0)"
    return f"{sql} AS {alias}" if alias else sql


def four_type_token_total(row: Any) -> int:
    """The all-four-types total for one mapping-like row (or any object
    exposing the four columns as keys). Missing buckets read as 0."""
    getter = row.get if hasattr(row, "get") else (lambda k, d=0: getattr(row, k, d))
    return sum(int(getter(col, 0) or 0) for col in TOKEN_TYPE_COLUMNS)


def _attributes_dict(attributes: Any) -> dict[str, Any]:
    """Span attributes as a dict. Stored as a JSON string by some backends and
    as a dict by others; anything unparseable reads as empty, never raises."""
    if isinstance(attributes, dict):
        return attributes
    if isinstance(attributes, (str, bytes)):
        try:
            parsed = json.loads(attributes)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def call_identity(span_id: Any, session_id: Any = None, attributes: Any = None) -> str:
    """The identity of the underlying API call this span row describes.

    Prefers an explicit call id off the span's attributes; falls back to the
    row's own ``span_id``, which is unique per row and so preserves today's
    behaviour exactly for spans that carry no call id.
    """
    attrs = _attributes_dict(attributes)
    for key in CALL_ID_ATTRIBUTE_KEYS:
        value = attrs.get(key)
        if isinstance(value, str) and value:
            return f"{session_id or ''}|{value}"
    return f"{session_id or ''}|{span_id or ''}"


def ingest_source(attributes: Any) -> str:
    """Which ingest path recorded this observation.

    ``LIVE_INGEST_SOURCE`` when the span carries no source attribute — the
    live receive path never stamps one, every backfill adapter does.
    """
    value = _attributes_dict(attributes).get(INGEST_SOURCE_ATTRIBUTE)
    return value if isinstance(value, str) and value else LIVE_INGEST_SOURCE


def call_fingerprint(
    session_id: Any,
    model: Any,
    input_tokens: Any,
    output_tokens: Any,
    cache_tokens: Any,
    cache_write_tokens: Any,
) -> str:
    """The billed shape of one call, as every observer of it sees it.

    Two observations of one API call agree on all six of these; a stamped id
    is unavailable from at least one side of the Claude Code pair, and these
    columns are. Derived from stored columns rather than from an attribute, so
    it reads rows written before any ingest path stamped anything.

    Deliberately NOT a claim of identity on its own — see the module docstring
    and ``duplicate_budget``: it may only ever collapse ACROSS ingest sources.
    """
    parts = "|".join(
        str(part) for part in (
            session_id or "", model or "",
            int(input_tokens or 0), int(output_tokens or 0),
            int(cache_tokens or 0), int(cache_write_tokens or 0),
        )
    )
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]


def duplicate_budget(
    stored_by_source: Mapping[str, int],
    own_source: str,
    already_suppressed: int = 0,
) -> int:
    """How many further observations of one call ``own_source`` may suppress.

    ``stored_by_source`` counts the observations of a single call already in
    the store, per ingest source. A source may suppress as many observations
    as the most complete OTHER source recorded, minus what it has already
    stored itself and minus what it has already suppressed this run; anything
    beyond that is a genuinely new call and must be KEPT.

    Never negative, and the ``already_suppressed`` term is what keeps it that
    way across a run: a suppressed observation is never stored, so without it
    the same budget would be spent again on the next arrival and a real second
    call would be dropped. Dropping a real call under-reports spend, which is
    the failure direction this whole seam exists to avoid, so the rule leans
    to keeping: an observer that recorded MORE of a call than any other saw
    work nobody else did, which is evidence of a gap elsewhere, never of a
    duplicate here.
    """
    from_others = max(
        (n for source, n in stored_by_source.items() if source != own_source),
        default=0,
    )
    own = stored_by_source.get(own_source, 0) + already_suppressed
    return max(from_others - own, 0)


def dedupe_by_call_identity(
    rows: Iterable[Sequence[Any]], *, identity_index: int = 0,
) -> list[Sequence[Any]]:
    """Collapse rows describing the same call, LAST WINS.

    Last-wins mirrors ``core.usage.session_usage``: when one call is observed
    more than once, the later observation is the finalized one. Order among
    distinct calls is preserved (first appearance), so callers that report
    counts get a stable result.
    """
    latest: dict[Any, Sequence[Any]] = {}
    order: list[Any] = []
    for row in rows:
        key = row[identity_index]
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]
