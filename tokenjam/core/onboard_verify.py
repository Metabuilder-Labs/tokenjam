"""
First-signal onboarding verification (#80).

Two onboarding personas can "successfully" finish setup yet never emit a single
span, with no signal that anything is wrong:

  - **SDK**: a bare ``tj onboard`` ends by printing a ``patch_anthropic()`` +
    ``@watch()`` snippet. The SDK is deliberately fail-open, so a typo'd
    ``agent_id``, a missing patch call, or a dead daemon produces zero spans
    silently.
  - **Claude Code / Codex**: telemetry only starts after the user restarts the
    agent runtime. If they never restart, ingest never begins and nothing tells
    them.

This module provides the shared, side-effect-light polling primitive used by
``tj onboard --verify`` (and the post-onboard prompt) to distinguish "wired and
receiving" from "configured but silent". It is import-safe from both the CLI and
tests, and takes no DuckDB write lock — it reads through whichever backend is
resolved by :func:`open_read_backend` (the running daemon over HTTP when present,
a direct read otherwise).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol

from tokenjam.core.models import TraceFilters

# Poll cadence defaults. ~60s total is long enough for a human to restart Claude
# Code or run `tj ping` in another terminal, short enough to not feel hung.
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_INTERVAL_S = 2.0


class _ReadBackend(Protocol):
    """The single method the poller needs — both DuckDBBackend and ApiBackend
    satisfy this."""

    def get_traces(self, filters: TraceFilters) -> list: ...


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a first-span poll.

    ``confirmed`` is True when at least one span arrived after ``since``.
    ``error`` is set only when the read itself failed (e.g. the DB went away
    mid-poll); a clean "nothing arrived within the timeout" is ``confirmed=False``
    with ``error=None``.
    """

    confirmed: bool
    elapsed_s: float
    first_trace_id: str | None = None
    error: str | None = None


def poll_for_first_span(
    backend: _ReadBackend,
    since: datetime,
    *,
    agent_id: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> VerifyResult:
    """Poll ``backend`` until a trace whose spans started at/after ``since``
    appears, or ``timeout_s`` elapses.

    ``agent_id`` narrows the poll to a single source when the caller knows it;
    callers that can't reliably predict the ingested ``agent_id`` (e.g. Claude
    Code, whose spans carry a ``service.name``-derived id that differs from the
    ``claude-code-<project>`` config key) should leave it ``None`` and rely on
    the ``since`` time bound.

    ``sleep`` / ``monotonic`` are injectable so tests never wait real seconds.
    """
    start = monotonic()
    while True:
        try:
            traces = backend.get_traces(
                TraceFilters(since=since, agent_id=agent_id, limit=1)
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't crash onboarding
            return VerifyResult(False, monotonic() - start, error=str(exc))

        if traces:
            return VerifyResult(
                True, monotonic() - start, first_trace_id=traces[0].trace_id
            )

        if monotonic() - start >= timeout_s:
            return VerifyResult(False, monotonic() - start)

        sleep(interval_s)


def not_confirmed_cause(persona: str) -> str:
    """Return the most likely reason no span arrived, phrased for the persona.

    ``persona`` is one of ``"sdk"``, ``"claude_code"``, ``"codex"``; anything
    else falls back to a generic message that covers both failure modes.
    """
    if persona == "claude_code":
        return (
            "Claude Code only starts sending telemetry after a restart. Open a "
            "new terminal and launch `claude`, then re-run `tj doctor`. If you "
            "already restarted, check the OTLP wiring with `tj doctor`."
        )
    if persona == "codex":
        return (
            "Codex only starts sending telemetry after a restart. Restart Codex, "
            "then re-run `tj doctor`. If you already restarted, check the OTLP "
            "wiring with `tj doctor`."
        )
    if persona == "sdk":
        return (
            "Make sure patch_anthropic()/@watch() actually run in your agent and "
            "the daemon is reachable. Run `tj ping` to emit a labeled test span, "
            "then `tj status`."
        )
    return (
        "If you use Claude Code / Codex, restart it so telemetry starts. If you "
        "use the SDK, confirm patch_*()/@watch() run and the daemon is up — try "
        "`tj ping`."
    )


def open_read_backend(config) -> tuple[object | None, str | None, str | None]:
    """Resolve a read-only path to spans without competing for the DuckDB write
    lock.

    Returns ``(backend, mode, error)`` where ``mode`` is ``"api"`` or
    ``"direct"`` on success (``error`` None), or ``(None, None, error)`` when
    neither path is available.

    The running daemon is tried **first** (a plain HTTP GET, no lock): in the
    default post-onboard state ``tj serve`` holds the DuckDB write lock, and the
    poller must never take a competing lock (DuckDB is single-writer). Only when
    no daemon answers do we open the file directly — the pre-daemon / --no-daemon
    case, where nothing else holds the lock.
    """
    from tokenjam.core.api_backend import probe_api

    api_key = config.api.auth.api_key if config.api.auth.enabled else None
    api = probe_api(config.api.host, config.api.port, api_key)
    if api is not None:
        return api, "api", None

    from tokenjam.core.db import open_db

    try:
        return open_db(config.storage), "direct", None
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "lock" in msg or "already open" in msg or "i/o error" in msg:
            return (
                None,
                None,
                "the database is locked (is tj serve running?) and its API isn't "
                f"reachable at http://{config.api.host}:{config.api.port}",
            )
        return None, None, str(exc)
