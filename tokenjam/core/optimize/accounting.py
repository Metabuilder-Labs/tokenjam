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

Today a span only carries an explicit call id when an ingest path stamps one;
without it the fallback is the row's own ``span_id``, which is what the store
has always keyed on. That makes this a seam, not a rewrite: adopting it changes
no current number, and the day the ingest paths agree on a call id, every
figure routed through here becomes single-counted with no further edits. The
ingest-side deduplication is separate work on its own track; nothing here
modifies how spans are written.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

#: The four billable token columns on a span row, in canonical order.
TOKEN_TYPE_COLUMNS: tuple[str, ...] = (
    "input_tokens", "output_tokens", "cache_tokens", "cache_write_tokens",
)

#: Attribute keys, in precedence order, that name the underlying API call.
#: First non-empty one wins. ``gen_ai.response.id`` is the provider's own id
#: for the response; ``tj.call_id`` is the internal stamp.
CALL_ID_ATTRIBUTE_KEYS: tuple[str, ...] = ("tj.call_id", "gen_ai.response.id")


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
