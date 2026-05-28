"""
Test the SDK HttpTransport's 401 fail-fast behavior (#68 §2).

Before this fix, a 401 from tj serve was silently retried 3× with
exponential backoff (~6s of stalling per send), then the spans got
buffered. Users hitting a config-secret mismatch saw their spans
disappear and only a single line of warning output.

After the fix:
- 401 is detected, logged at ERROR level (loud), and the spans drop
  immediately (no buffering — they'd never succeed).
- dropped_auth_failures counter increments so `tj doctor` can later
  surface cumulative loss to the user.
- No 6s stall — fail fast.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from tokenjam.core.config import ApiConfig, SecurityConfig, TjConfig
from tokenjam.sdk.transport import HttpTransport


@pytest.fixture
def transport() -> HttpTransport:
    config = TjConfig(
        version="1",
        api=ApiConfig(host="127.0.0.1", port=7391),
        security=SecurityConfig(ingest_secret="local-secret-aabbcc1122"),
    )
    return HttpTransport(config)


def test_401_fails_fast_and_clears_buffer(transport):
    """A 401 response should drop the spans immediately, no retry."""
    spans = [{"span_id": "s1"}, {"span_id": "s2"}]

    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("tokenjam.sdk.transport.httpx.post", return_value=mock_resp) as mock_post, \
         patch("tokenjam.sdk.transport.time.sleep") as mock_sleep:
        result = transport.send(spans)

    assert result is False
    # Critical: only one POST attempt (no retry). Pre-fix would have been 3.
    assert mock_post.call_count == 1, (
        "401 should not be retried — it won't change on retry"
    )
    # Critical: no exponential backoff sleep.
    assert mock_sleep.call_count == 0, (
        "401 should fail fast without waiting through backoff"
    )
    # Spans are dropped (cleared), not buffered for next attempt.
    assert transport._buffer == []
    # Counter incremented.
    assert transport.dropped_auth_failures == 2


def test_401_counter_accumulates_across_calls(transport):
    """Successive 401s accumulate the dropped count."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("tokenjam.sdk.transport.httpx.post", return_value=mock_resp), \
         patch("tokenjam.sdk.transport.time.sleep"):
        transport.send([{"span_id": "a"}])
        transport.send([{"span_id": "b"}, {"span_id": "c"}])
        transport.send([{"span_id": "d"}])

    assert transport.dropped_auth_failures == 4


def test_non_401_errors_still_retry_and_buffer(transport):
    """Non-401 4xx/5xx still get the original retry-and-buffer treatment."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503

    with patch("tokenjam.sdk.transport.httpx.post", return_value=mock_resp) as mock_post, \
         patch("tokenjam.sdk.transport.time.sleep"):
        result = transport.send([{"span_id": "x"}])

    assert result is False
    # Full 3 attempts (backoff between).
    assert mock_post.call_count == 3
    # Spans stay buffered (not dropped) — could succeed if server recovers.
    assert transport._buffer == [{"span_id": "x"}]
    # Auth-failure counter NOT incremented for non-401 errors.
    assert transport.dropped_auth_failures == 0


def test_success_path_unchanged(transport):
    """Happy path: 2xx clears buffer and returns True."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("tokenjam.sdk.transport.httpx.post", return_value=mock_resp) as mock_post, \
         patch("tokenjam.sdk.transport.time.sleep"):
        result = transport.send([{"span_id": "ok"}])

    assert result is True
    assert mock_post.call_count == 1
    assert transport._buffer == []
    assert transport.dropped_auth_failures == 0


def test_401_logs_secret_fingerprint(transport, caplog):
    """The error message should include a truncated secret fingerprint."""
    import logging
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("tokenjam.sdk.transport.httpx.post", return_value=mock_resp), \
         patch("tokenjam.sdk.transport.time.sleep"), \
         caplog.at_level(logging.ERROR, logger="tokenjam.sdk.transport"):
        transport.send([{"span_id": "z"}])

    assert any(
        "local-se..." in r.message and "401" in r.message
        for r in caplog.records
    ), "error log should mention the truncated SDK secret + 401"
