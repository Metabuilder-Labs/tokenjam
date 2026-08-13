"""``_dict_to_span`` / ``get_traces`` reconstruct NormalizedSpan/TraceRecord from
the HTTP-API's JSON, which is the ApiBackend read path used when `tj serve`
holds the DuckDB write lock.

Both `NormalizedSpan.start_time` and `TraceRecord.start_time` are declared
non-Optional (models.py) because ingest already rejects any record with no
observed time rather than substituting a default or leaving it unset. A
`cast(datetime, ... or None)` here used to silently smuggle a `None` past that
contract when the wire payload was missing `start_time` — which suppressed the
exact signal that would have caught unguarded downstream dereferences
(`core/cost.py`, the cache-efficacy and model-downgrade analyzers). These
tests pin the replacement contract: a genuinely present `start_time` still
round-trips, and a missing one now raises instead of laundering a `None`
through as if it were a `datetime`.
"""
from __future__ import annotations

import pytest

from tokenjam.core.api_backend import _dict_to_span, _require_start_time


def _span_dict(**overrides):
    d = {
        "span_id": "sp-1",
        "trace_id": "tr-1",
        "name": "llm.completion",
        "kind": "internal",
        "status_code": "ok",
        "start_time": "2026-03-14T12:00:00+00:00",
    }
    d.update(overrides)
    return d


def test_dict_to_span_parses_a_present_start_time():
    span = _dict_to_span(_span_dict())
    assert span.start_time.isoformat() == "2026-03-14T12:00:00+00:00"


def test_dict_to_span_raises_when_start_time_is_missing():
    """A malformed/incomplete API response must fail loudly, not hand a
    `None` to a field the type system promises is always a real `datetime`."""
    with pytest.raises(ValueError, match="start_time"):
        _dict_to_span(_span_dict(start_time=None))


def test_dict_to_span_raises_when_start_time_key_is_absent():
    d = _span_dict()
    del d["start_time"]
    with pytest.raises(ValueError, match="start_time"):
        _dict_to_span(d)


def test_require_start_time_identifies_the_offending_record():
    with pytest.raises(ValueError, match="sp-1"):
        _require_start_time({"span_id": "sp-1", "start_time": None})
