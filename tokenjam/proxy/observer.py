"""Observe-only recorder for proxy gate decisions (#219).

Every request through the proxy produces an inspectable record of what the gate
decided (and, on the policy path, what a policy *would* do — nothing yet, this
is suggest mode). Records are logged and kept in a small in-memory ring so the
``tj proxy status`` command and tests can inspect recent decisions without any
DB coupling. This is deliberately lightweight: the proxy never blocks on
recording, and a recording failure must never affect pass-through.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque

from tokenjam.proxy.gate import GateDecision
from tokenjam.utils.time_parse import utcnow

logger = logging.getLogger("tokenjam.proxy")


@dataclass(frozen=True)
class ProxyObservation:
    """One observed request: the gate decision + request shape (no payload)."""
    ts:           str
    method:       str
    path:         str          # request URL path, e.g. /v1/messages
    provider:     str
    pricing_mode: str
    decision:     str          # observe_only | policy
    reason:       str
    forwarded:    bool         # always True in suggest mode
    suggest_only: bool = True  # suggest mode enforces nothing


class ProxyObserver:
    """Keeps the last ``maxlen`` observations and logs each one."""

    def __init__(self, maxlen: int = 256) -> None:
        self._ring: Deque[ProxyObservation] = deque(maxlen=maxlen)

    def record(self, *, method: str, path: str, decision: GateDecision,
               forwarded: bool = True) -> ProxyObservation:
        obs = ProxyObservation(
            ts=utcnow().isoformat(),
            method=method,
            path=path,
            provider=decision.provider,
            pricing_mode=decision.pricing_mode,
            decision=decision.path,
            reason=decision.reason,
            forwarded=forwarded,
        )
        self._ring.append(obs)
        # Suggest mode: log what the gate decided; never raise out of recording.
        try:
            logger.info(
                "proxy %s %s provider=%s mode=%s decision=%s reason=%s forwarded=%s",
                method, path, obs.provider, obs.pricing_mode, obs.decision,
                obs.reason, forwarded,
            )
        except Exception:  # noqa: BLE001 — recording must never break pass-through
            pass
        return obs

    @property
    def observations(self) -> list[ProxyObservation]:
        return list(self._ring)

    def as_dicts(self) -> list[dict]:
        return [asdict(o) for o in self._ring]
