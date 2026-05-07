"""
Span transport layer. Delivers spans via HTTP POST to the local tj serve
instance (used when tj serve is a separate process, or by the TypeScript SDK).

The transport is initialised once when the first @watch() call is made.
Subsequent calls reuse the same transport.
"""
from __future__ import annotations

import logging
import time

import httpx

from tj.core.config import TjConfig

logger = logging.getLogger(__name__)

_MAX_BUFFER = 1000
_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds


class HttpTransport:
    """
    Posts spans to POST /api/v1/spans on the local tj serve instance.

    Buffers up to 1000 spans if tj serve is not reachable.
    Retries with exponential backoff (max 3 attempts, 2s base delay).
    Drops buffered spans on process exit with a log warning.
    """

    def __init__(self, config: TjConfig):
        self.endpoint = (
            f"http://{config.api.host}:{config.api.port}/api/v1/spans"
        )
        self.secret = config.security.ingest_secret
        self._buffer: list[dict] = []

    def send(self, spans: list[dict]) -> bool:
        """
        POST spans to tj serve.
        Returns True on success, False on failure (spans are buffered).
        """
        # Add new spans to buffer
        self._buffer.extend(spans)
        if len(self._buffer) > _MAX_BUFFER:
            dropped = len(self._buffer) - _MAX_BUFFER
            self._buffer = self._buffer[-_MAX_BUFFER:]
            logger.warning("Dropped %d buffered spans (buffer full)", dropped)

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.secret:
            headers["Authorization"] = f"Bearer {self.secret}"

        payload = self._buffer.copy()
        for attempt in range(_MAX_RETRIES):
            try:
                resp = httpx.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=5.0,
                )
                if resp.status_code < 300:
                    self._buffer.clear()
                    return True
                logger.warning(
                    "tj serve returned %d on attempt %d",
                    resp.status_code, attempt + 1,
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                logger.debug(
                    "tj serve unreachable (attempt %d/%d): %s",
                    attempt + 1, _MAX_RETRIES, exc,
                )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BASE_DELAY * (2 ** attempt))

        logger.warning(
            "Failed to send %d spans after %d attempts; buffered for retry",
            len(payload), _MAX_RETRIES,
        )
        return False

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)

    def flush(self) -> None:
        """Attempt to send all buffered spans. Called on shutdown."""
        if not self._buffer:
            return
        if not self.send([]):
            logger.warning(
                "Dropping %d buffered spans on shutdown", len(self._buffer),
            )
            self._buffer.clear()
