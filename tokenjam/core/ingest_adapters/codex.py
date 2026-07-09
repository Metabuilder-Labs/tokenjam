"""
Codex CLI on-disk session (rollout) → TokenJam ingest adapter.

The OpenAI Codex CLI persists every session as a newline-delimited JSON
"rollout" file under::

    ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl

Each line is a `RolloutLine`: ``{"timestamp": "<iso8601>", "type": "<kind>",
"payload": {...}}`` (the codex-rs ``RolloutItem`` enum is serialized with
``tag = "type"`` + ``content = "payload"``). The line kinds we consume:

  - ``session_meta`` — carries ``session_id`` (or ``id`` thread id on newer
    builds), ``timestamp``, ``cwd``, ``cli_version``, ``model_provider``. One
    per file, the first line.
  - ``turn_context`` — carries ``model`` (the effective model for the turns
    that follow). We track the latest one as the "current model".
  - ``event_msg`` with an inner ``{"type": "token_count", "info": {...}}`` —
    the usage update. ``info.last_token_usage`` is the *delta* for the turn
    just completed; ``info.total_token_usage`` is the running cumulative total.
    We emit one LLM span per ``token_count`` event, using the per-turn delta so
    the summed session cost matches the cumulative total (no double-count).
    ``TokenUsage`` fields: ``input_tokens``, ``cached_input_tokens``,
    ``output_tokens``, ``reasoning_output_tokens``, ``total_tokens``.
  - ``response_item`` with an inner ``{"type": "function_call", ...}`` — a tool
    invocation (``name``, ``call_id``). Becomes a tool span.

This mirrors ``core.backfill.ingest_claude_code``:
  - **Deterministic span IDs** (hash of session_id + a stable per-line key) so
    re-running ingests no duplicates.
  - **Plan tier** stamped from ``config.budgets["openai"].plan`` — Codex is
    always OpenAI — matching the live ingest path so backfilled sessions aren't
    all ``"unknown"`` (#176).
  - **Cost recomputed** from ``pricing/models.toml`` (the rollout has no cost).
  - Session totals reconciled from the inserted spans after ingest.

Codex hardcodes ``service.name=codex_exec`` in its binary, so every live-ingested
Codex span lands under the ``codex_exec`` agent id regardless of project. We use
the same agent id here so backfilled sessions attribute to the same agent.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tokenjam.core.backfill import _existing_span_ids
from tokenjam.core.cost import calculate_cost
from tokenjam.core.models import (
    NormalizedSpan,
    SessionRecord,
    SpanKind,
    SpanStatus,
)

logger = logging.getLogger(__name__)


CODEX_SESSIONS_ROOT = Path.home() / ".codex" / "sessions"

# The `attributes.source` tag every Codex backfill span carries. Scopes any
# future stale-scheme reconciliation to backfill-sourced rows only.
_CODEX_SOURCE = "backfill.codex"

# Codex always bills OpenAI — its plan tier lives under [budget.openai] in
# config, and its live spans set provider/billing_account="openai".
_CODEX_PROVIDER = "openai"

# Codex hardcodes service.name=codex_exec; the live logs path attributes every
# span to it. Mirror that so backfill and live ingest agree on the agent id.
_CODEX_AGENT_ID = "codex_exec"


def _plan_tier_for_provider(config, provider: str) -> str:
    """Resolve plan_tier from config the same way the live ingest path does
    (`IngestPipeline._resolve_plan_tier`): `config.budgets[provider].plan`,
    falling back to "unknown". Mirrors `core.backfill._plan_tier_for_provider`.
    """
    if config is None:
        return "unknown"
    budgets = getattr(config, "budgets", None) or {}
    bcfg = budgets.get(provider)
    if bcfg is None or not getattr(bcfg, "plan", None):
        return "unknown"
    return bcfg.plan


@dataclass
class ParsedCodexSession:
    session_id: str
    agent_id: str
    started_at: datetime
    ended_at: datetime
    cwd: str | None
    spans: list[NormalizedSpan]
    total_input_tokens: int
    total_output_tokens: int
    total_cache_tokens: int
    total_cost_usd: float
    tool_call_count: int


# --- ID derivation helpers ---------------------------------------------------

def _det_id(*parts: str, length: int = 16) -> str:
    """Deterministic hex ID derived from the given parts."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return h[:length]


def _trace_id_for(session_id: str) -> str:
    """One trace per Codex session, matching the live logs path
    (`routes/logs.py._trace_id_from_session` groups by session/conversation)."""
    return _det_id("codex-trace", session_id, length=32)


def _span_id_for_llm(session_id: str, turn_key: str) -> str:
    return _det_id("codex-llm", session_id, turn_key)


def _span_id_for_tool(session_id: str, call_id: str) -> str:
    return _det_id("codex-tool", session_id, call_id)


def _parse_ts(value: Any) -> datetime | None:
    """Parse a Codex rollout ISO-8601 timestamp string to an aware UTC datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        raw = value
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _as_int(value: Any) -> int:
    """Coerce a rollout token field to a non-negative int; 0 on anything odd."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n >= 0 else 0


# --- Rollout parser ----------------------------------------------------------

def parse_codex_rollout(path: Path) -> ParsedCodexSession | None:
    """Parse a single Codex rollout JSONL file into a ParsedCodexSession.

    Returns None when the file has no usable session id or no token-bearing
    turns (e.g. a session that ended before the first model response).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None

    session_id: str | None = None
    cwd: str | None = None
    current_model: str | None = None
    earliest: datetime | None = None
    latest: datetime | None = None

    # Deterministic dedup within the file. A rollout is append-only and never
    # replays a line, but keying by span_id keeps ingest idempotent across
    # re-runs regardless.
    spans_by_id: dict[str, NormalizedSpan] = {}
    # Monotonic per-file index so successive token_count turns get distinct,
    # reproducible span ids even without an explicit turn id in the payload.
    turn_index = 0

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue

        rtype = rec.get("type")
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        ts = _parse_ts(rec.get("timestamp"))
        if ts is not None:
            if earliest is None or ts < earliest:
                earliest = ts
            if latest is None or ts > latest:
                latest = ts

        if rtype == "session_meta":
            # `session_id` on current builds; older/renamed builds carried the
            # thread id under `id`. Either is a stable per-session identifier.
            if session_id is None:
                session_id = payload.get("session_id") or payload.get("id")
            if cwd is None:
                cwd = payload.get("cwd")
            continue

        if rtype == "turn_context":
            model = payload.get("model")
            if isinstance(model, str) and model:
                current_model = model
            continue

        if rtype == "event_msg" and payload.get("type") == "token_count":
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            # `last_token_usage` is the delta for the turn that just completed;
            # summing deltas across turns reconstructs the session total without
            # double-counting the cumulative `total_token_usage`.
            usage = info.get("last_token_usage")
            if not isinstance(usage, dict):
                continue

            input_tokens = _as_int(usage.get("input_tokens"))
            cached = _as_int(usage.get("cached_input_tokens"))
            output_tokens = _as_int(usage.get("output_tokens"))
            reasoning = _as_int(usage.get("reasoning_output_tokens"))
            # Codex reports cached_input_tokens as a SUBSET of input_tokens
            # (matching the Responses API `usage.input_tokens_details`), so the
            # billable non-cached input is input - cached. Reasoning tokens are
            # billed at the output rate, so fold them into output for costing.
            non_cached_input = max(input_tokens - cached, 0)
            output_total = output_tokens + reasoning

            if input_tokens == 0 and output_total == 0 and cached == 0:
                continue

            sid = session_id or path.stem
            turn_index += 1
            turn_key = str(turn_index)
            span_id = _span_id_for_llm(sid, turn_key)
            trace_id = _trace_id_for(sid)
            start_time = ts or datetime.now(tz=timezone.utc)

            # OpenAI automatic prompt caching: cached_input_tokens are read-hits
            # billed at the (lower) cache-read rate; there is no separate
            # cache-creation charge (mirrors the live logs path, #93).
            cost = calculate_cost(
                provider=_CODEX_PROVIDER,
                model=current_model or "unknown",
                input_tokens=non_cached_input,
                output_tokens=output_total,
                cache_read_tokens=cached,
                cache_write_tokens=0,
            )

            spans_by_id[span_id] = NormalizedSpan(
                span_id=span_id,
                trace_id=trace_id,
                name="gen_ai.llm.call",
                kind=SpanKind.CLIENT,
                status_code=SpanStatus.OK,
                start_time=start_time,
                end_time=start_time,
                duration_ms=None,
                agent_id=_CODEX_AGENT_ID,
                session_id=sid,
                provider=_CODEX_PROVIDER,
                model=current_model,
                input_tokens=non_cached_input,
                output_tokens=output_total,
                cache_tokens=cached,
                cache_write_tokens=0,
                cost_usd=cost,
                request_type="completion",
                conversation_id=sid,
                attributes={"source": _CODEX_SOURCE},
                billing_account=_CODEX_PROVIDER,
            )
            continue

        if rtype == "response_item" and payload.get("type") == "function_call":
            sid = session_id or path.stem
            call_id = payload.get("call_id") or _det_id(
                "codex-tool-fallback", sid, str(line_no)
            )
            tool_span_id = _span_id_for_tool(sid, call_id)
            tool_name = payload.get("name") or "unknown"
            start_time = ts or datetime.now(tz=timezone.utc)
            spans_by_id[tool_span_id] = NormalizedSpan(
                span_id=tool_span_id,
                trace_id=_trace_id_for(sid),
                name="gen_ai.tool.call",
                kind=SpanKind.INTERNAL,
                status_code=SpanStatus.OK,
                start_time=start_time,
                end_time=start_time,
                duration_ms=None,
                agent_id=_CODEX_AGENT_ID,
                session_id=sid,
                tool_name=tool_name,
                conversation_id=sid,
                attributes={"source": _CODEX_SOURCE},
            )
            continue

    if not spans_by_id or session_id is None:
        return None

    spans = list(spans_by_id.values())
    total_input = total_output = total_cache = tool_count = 0
    total_cost = 0.0
    for s in spans:
        if s.name == "gen_ai.tool.call":
            tool_count += 1
            continue
        total_input += s.input_tokens or 0
        total_output += s.output_tokens or 0
        total_cache += s.cache_tokens or 0
        total_cost += s.cost_usd or 0.0

    started_at = earliest or datetime.now(tz=timezone.utc)
    ended_at = latest or started_at

    return ParsedCodexSession(
        session_id=session_id,
        agent_id=_CODEX_AGENT_ID,
        started_at=started_at,
        ended_at=ended_at,
        cwd=cwd,
        spans=spans,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cache_tokens=total_cache,
        total_cost_usd=round(total_cost, 8),
        tool_call_count=tool_count,
    )


def iter_codex_sessions(
    root: Path | None = None,
    since: datetime | None = None,
) -> Iterator[ParsedCodexSession]:
    """Walk the Codex sessions directory and yield ParsedCodexSession objects.

    `since` filters out files whose mtime is before the cutoff (cheap
    pre-filter); the session's own end_time is re-checked per-file.
    """
    base = root or CODEX_SESSIONS_ROOT
    if not base.exists() or not base.is_dir():
        return

    for jsonl_path in sorted(base.rglob("rollout-*.jsonl")):
        try:
            if since is not None:
                mtime = datetime.fromtimestamp(
                    jsonl_path.stat().st_mtime, tz=timezone.utc
                )
                if mtime < since:
                    continue
        except OSError:
            continue
        parsed = parse_codex_rollout(jsonl_path)
        if parsed is None:
            continue
        if since is not None and parsed.ended_at < since:
            continue
        yield parsed


def session_record_from_parsed(
    parsed: ParsedCodexSession, plan_tier: str = "unknown",
) -> SessionRecord:
    return SessionRecord(
        session_id=parsed.session_id,
        agent_id=parsed.agent_id,
        started_at=parsed.started_at,
        ended_at=parsed.ended_at,
        conversation_id=parsed.session_id,
        status="completed",
        total_cost_usd=parsed.total_cost_usd,
        input_tokens=parsed.total_input_tokens,
        output_tokens=parsed.total_output_tokens,
        cache_tokens=parsed.total_cache_tokens,
        tool_call_count=parsed.tool_call_count,
        error_count=0,
        plan_tier=plan_tier,
    )


# --- Ingest -----------------------------------------------------------------

def ingest_codex(
    db,
    *,
    root: Path | None = None,
    since: datetime | None = None,
    config=None,
) -> dict[str, int]:
    """Ingest Codex CLI rollout sessions into the storage backend.

    `db` is a DuckDBBackend (or compatible). Writes are idempotent: spans whose
    span_id already exists are skipped, so re-running never duplicates.

    `config` (a TjConfig) supplies the declared plan tier so backfilled sessions
    carry the same `plan_tier` the live ingest path sets (#176). When None or no
    plan is configured, sessions fall back to "unknown".

    Returns a summary dict:
    ``{"sessions_seen", "sessions_written", "spans_written",
       "spans_skipped", "sessions_failed"}``.
    """
    plan_tier = _plan_tier_for_provider(config, _CODEX_PROVIDER)

    sessions_seen = 0
    sessions_written = 0
    spans_written = 0
    spans_skipped = 0
    sessions_failed = 0
    seen_session_ids: set[str] = set()

    conn = getattr(db, "conn", None)

    for parsed in iter_codex_sessions(root=root, since=since):
        sessions_seen += 1
        seen_session_ids.add(parsed.session_id)
        try:
            inserted = 0
            # Bulk idempotency (#433): partition the session's spans into
            # new-vs-existing with ONE chunked `WHERE span_id IN (...)` query
            # (via the shared `_existing_span_ids` helper the Claude Code path
            # uses) instead of a SELECT per span. When the backend has no `conn`
            # (e.g. InMemoryBackend), fall back to plain inserts and let the
            # PRIMARY KEY conflict skip duplicates.
            if conn is not None:
                existing = _existing_span_ids(conn, [s.span_id for s in parsed.spans])
            else:
                existing = set()
            for span in parsed.spans:
                if span.span_id in existing:
                    spans_skipped += 1
                    continue
                try:
                    db.insert_span(span)
                    inserted += 1
                except Exception:
                    spans_skipped += 1
            db.upsert_session(
                session_record_from_parsed(parsed, plan_tier=plan_tier)
            )
        except Exception as exc:
            sessions_failed += 1
            logger.warning("Failed to ingest Codex session %s: %s",
                           parsed.session_id, exc)
            continue

        spans_written += inserted
        if inserted > 0:
            sessions_written += 1

    # Reconcile each touched session row to the SUM of its spans, mirroring the
    # Claude Code backfill path — an idempotent re-run also repairs earlier rows.
    recompute = getattr(db, "recompute_session_totals_from_spans", None)
    if recompute is not None and seen_session_ids:
        recompute(sorted(seen_session_ids))

    return {
        "sessions_seen": sessions_seen,
        "sessions_written": sessions_written,
        "spans_written": spans_written,
        "spans_skipped": spans_skipped,
        "sessions_failed": sessions_failed,
    }


__all__ = [
    "CODEX_SESSIONS_ROOT",
    "ParsedCodexSession",
    "parse_codex_rollout",
    "iter_codex_sessions",
    "ingest_codex",
    "session_record_from_parsed",
]
