"""The CLI's data-access seam: one interface, two implementations.

Several analytics commands (`tj context`, `tj tokenmaxx`, `tj quota-audit`)
need computed reads that touch the raw DuckDB `attributes` column / per-session
token metadata — data the read-only HTTP shim (`ApiBackend`) can't expose at
that grain. DuckDB permits only one writer OR many readers across processes, so
a concurrent read-only connection alongside the `tj serve` write-lock is
impossible (it raises an IOException). The commands therefore have two modes:

  * **direct** — no daemon: compute against the CLI's own DuckDB connection;
  * **serve** — the daemon holds the write lock: route the compute through the
    daemon (which owns the direct connection) over the REST API.

Historically each command re-implemented that dispatch by *duck-typing* the
backend (`hasattr(db, "conn")` for direct, `hasattr(db, "fetch_*")` for the
shim), which drifted silently and left tokenmaxx / quota-audit with no serve
path at all. This module replaces the sniffing with one explicit `DataAccess`
interface: :class:`DirectDataAccess` and :class:`ServeDataAccess` satisfy the
same Protocol, so commands ask for the computed result and never branch on the
backend type. The long-term direction is daemon-owns-the-DB, CLI-direct is the
fallback — this seam makes that split explicit at one choke point
(:func:`resolve_data_access`) instead of scattered across commands.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import click

from tokenjam.core.context_diagnostic import (
    ContextDiagnostic,
    compute_context_diagnostic,
    diagnostic_from_dict,
)
from tokenjam.core.framing import (
    Framing,
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.core.optimize.analyzers.model_downgrade import audit_opus_quota
from tokenjam.core.optimize.types import OpusQuotaAudit, audit_from_dict
from tokenjam.utils.time_parse import parse_since, utcnow


@runtime_checkable
class DataAccess(Protocol):
    """Command-facing computed reads that need the raw database.

    Two implementations satisfy this — :class:`DirectDataAccess` (direct DuckDB
    connection) and :class:`ServeDataAccess` (routed through a running
    ``tj serve``). Each returns the fully-built dataclass plus its plan-tier
    :class:`~tokenjam.core.framing.Framing`, so the caller renders identically
    regardless of which backend produced the data.
    """

    def context_diagnostic(
        self, *, since: str, agent_id: str | None,
    ) -> tuple[ContextDiagnostic, Framing]: ...

    def quota_audit(
        self, *, since: str, agent_id: str | None,
    ) -> tuple[OpusQuotaAudit, Framing]: ...


class DirectDataAccess:
    """:class:`DataAccess` backed by a direct DuckDB connection (no daemon)."""

    def __init__(self, db: Any, config: Any) -> None:
        self._db = db
        self._config = config

    @property
    def _conn(self) -> Any:
        return self._db.conn

    def context_diagnostic(
        self, *, since: str, agent_id: str | None,
    ) -> tuple[ContextDiagnostic, Framing]:
        conn = self._conn
        since_dt = parse_since(since)
        until_dt = utcnow()
        # Capture flags come from the CLI's own config — the same source the
        # server path reads on the daemon side, so recurring-inclusion
        # detection is gated identically whether the daemon is up or not. They
        # affect only recurring/notes, never the token totals every consumer of
        # this seam renders.
        capture = getattr(self._config, "capture", None)
        diag = compute_context_diagnostic(
            conn,
            since_dt,
            until_dt,
            agent_id=agent_id,
            tool_inputs_captured=bool(getattr(capture, "tool_inputs", False)),
            prompts_captured=bool(getattr(capture, "prompts", False)),
            tool_outputs_captured=bool(getattr(capture, "tool_outputs", False)),
        )
        framing = _direct_framing(
            conn, self._config, diag.total_cost_usd, diag.total_tokens,
            diag.sessions, agent_id,
        )
        return diag, framing

    def quota_audit(
        self, *, since: str, agent_id: str | None,
    ) -> tuple[OpusQuotaAudit, Framing]:
        conn = self._conn
        since_dt = parse_since(since)
        until_dt = utcnow()
        window_days = max(
            (until_dt - since_dt).total_seconds() / 86400.0, 1.0 / 86400.0
        )
        audit = audit_opus_quota(conn, since_dt, until_dt, agent_id, window_days)
        framing = _direct_framing(
            conn, self._config, audit.actual_cost_usd, audit.opus_tokens,
            audit.opus_sessions, agent_id,
        )
        return audit, framing


class ServeDataAccess:
    """:class:`DataAccess` routed through a running ``tj serve`` (``ApiBackend``).

    The daemon owns the direct connection, so it computes the diagnostic / audit
    server-side and returns the serialized dataclass + a ``framing`` block; this
    reconstructs both so the command renders exactly as the direct path would.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    def context_diagnostic(
        self, *, since: str, agent_id: str | None,
    ) -> tuple[ContextDiagnostic, Framing]:
        payload = self._db.fetch_context_diagnostic(since=since, agent_id=agent_id)
        return diagnostic_from_dict(payload), _framing_from_payload(payload)

    def quota_audit(
        self, *, since: str, agent_id: str | None,
    ) -> tuple[OpusQuotaAudit, Framing]:
        payload = self._db.fetch_opus_quota_audit(since=since, agent_id=agent_id)
        return audit_from_dict(payload), _framing_from_payload(payload)


def _direct_framing(
    conn: Any, config: Any, total_cost_usd: float, total_tokens: int,
    sessions: int, agent_id: str | None,
) -> Framing:
    """Plan-tier framing for the window on the direct path.

    Plan determination is window-INDEPENDENT (#177) — the pricing mode is a
    property of the user's plan, not the selected window; only the totals are
    window-scoped. Mirrors the daemon-side framing in the API routes so the two
    paths render identical units + qualifier for the same DB.
    """
    mix = plan_determination_mix(conn, agent_id)
    return compute_framing(config, WindowSummary(
        total_cost_usd=total_cost_usd,
        total_tokens=total_tokens,
        sessions=sessions,
        plan_tier_mix=mix,
    ))


def _framing_from_payload(payload: dict) -> Framing:
    """Reconstruct a :class:`Framing` from a serialized ``framing`` block.

    ``Framing`` is a flat dataclass, so ``Framing(**block)`` round-trips
    ``to_dict()``. A missing block or a server-side schema drift degrades to the
    neutral default (raw token counts) rather than raising.
    """
    data = payload.get("framing") if isinstance(payload, dict) else None
    if not data:
        return Framing()
    try:
        return Framing(**data)
    except TypeError:
        return Framing()


def resolve_data_access(ctx: click.Context) -> DataAccess:
    """Pick the :class:`DataAccess` implementation for the current invocation.

    The single choke point that resolves direct-vs-serve, so no command
    branches on the backend type. ``ctx.obj["db"]`` is either a direct backend
    (a ``.conn``-bearing DuckDBBackend) or an ``ApiBackend`` (the daemon-holds-
    the-lock fallback wired up in ``cli/main.py``). Raises a clean
    ``ClickException`` — never a traceback — when neither is available.
    """
    obj = ctx.obj or {}
    db = obj.get("db")
    config = obj.get("config")
    if db is None or config is None:
        raise click.ClickException("this command requires a database connection.")

    # Import here to keep the module importable without httpx at import time.
    from tokenjam.core.api_backend import ApiBackend

    if isinstance(db, ApiBackend):
        return ServeDataAccess(db)
    if getattr(db, "conn", None) is not None:
        return DirectDataAccess(db, config)
    raise click.ClickException(
        "this command needs a direct DuckDB connection or a running tj serve "
        "at the configured api.{host,port}."
    )
