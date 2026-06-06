"""
Backfill: parse historical agent session logs into NormalizedSpan objects.

Currently supports Claude Code on-disk JSONL files at ~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl.
Each file contains one JSON object per line; relevant types are:
  - "assistant": message.model + message.usage.{input_tokens, output_tokens,
                 cache_read_input_tokens, cache_creation_input_tokens}.
                 message.content may contain {"type":"tool_use","name":...,"id":...}.
  - "user":      string content (user prompt) or list with tool_result items
                 (we don't need tool_result for v1 analyzers — tool_use is enough).

Other agent log formats (Codex, etc.) plug in by adding a new iter_* function
that yields the same (BackfillSession, list[NormalizedSpan]) tuples.

Cost is recomputed from pricing/models.toml — the on-disk format has no cost_usd.
Span IDs are deterministic (hash of session_id + assistant uuid / tool_use id) so
backfill is idempotent: re-running ingests no duplicates.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tokenjam.core.cost import calculate_cost
from tokenjam.core.models import (
    NormalizedSpan,
    SessionRecord,
    SpanKind,
    SpanStatus,
)

logger = logging.getLogger(__name__)


CLAUDE_CODE_PROJECTS_ROOT = Path.home() / ".claude" / "projects"

# Claude Code always bills Anthropic — its plan tier lives under
# [budget.anthropic] in config.
_CLAUDE_CODE_PROVIDER = "anthropic"


def _plan_tier_for_provider(config, provider: str) -> str:
    """Resolve plan_tier from config the same way the live ingest path does
    (`IngestPipeline._resolve_plan_tier`): `config.budgets[provider].plan`,
    falling back to "unknown".

    The Claude Code backfill bypasses `IngestPipeline`, so before #176 it
    created every session with the default `plan_tier="unknown"` even when
    config declared a plan — a split-brain state where `tj tokenmaxx` (reads
    config) and `tj optimize` (reads sessions) disagreed.
    """
    if config is None:
        return "unknown"
    budgets = getattr(config, "budgets", None) or {}
    bcfg = budgets.get(provider)
    if bcfg is None or not getattr(bcfg, "plan", None):
        return "unknown"
    return bcfg.plan


@dataclass
class BackfillResult:
    # `sessions_seen` counts conversation *files* parsed in the window — Claude
    # Code writes many JSONL files (continuations, sidechains) that can share
    # one sessionId, so this is NOT the number of rows that land in the
    # `sessions` table. Use the distinct-session counts below for that (#238).
    sessions_seen: int = 0
    sessions_ingested: int = 0
    spans_ingested: int = 0
    spans_skipped_existing: int = 0
    files_failed: int = 0
    earliest: datetime | None = None
    latest: datetime | None = None
    total_cost_usd: float = 0.0
    project_count: int = 0
    sample_errors: list[str] = field(default_factory=list)
    # Distinct session_ids seen in the window, and the subset that received at
    # least one newly-inserted span this run. These match the `sessions` table
    # (which is upserted by session_id), so the summary can report
    # new / already-present / total honestly instead of new-only (#238).
    seen_session_ids: set[str] = field(default_factory=set)
    new_session_ids: set[str] = field(default_factory=set)

    @property
    def conversations_seen(self) -> int:
        """Conversation files parsed (alias for sessions_seen, clearer label)."""
        return self.sessions_seen

    @property
    def sessions_total(self) -> int:
        """Distinct sessions in the window — matches the `sessions` table."""
        return len(self.seen_session_ids)

    @property
    def sessions_new(self) -> int:
        """Distinct sessions that gained at least one new span this run."""
        return len(self.new_session_ids)

    @property
    def sessions_existing(self) -> int:
        """Distinct sessions already fully present before this run."""
        return self.sessions_total - self.sessions_new


@dataclass
class ParsedSession:
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
    """One trace per session/conversation.

    Keyed on the session id alone (NOT per assistant message) so a whole
    conversation is a single trace with its LLM calls and tool calls as
    children — the Traces view then shows real session-level waterfalls
    instead of ~1.5-span per-message fragments (#243). This matches the live
    Claude Code log path (`routes/logs.py._trace_id_from_session`), which
    already groups by session.
    """
    return _det_id("trace", session_id, length=32)


def _span_id_for_assistant(session_id: str, message_uuid: str) -> str:
    return _det_id("llm", session_id, message_uuid)


def _span_id_for_tool(session_id: str, tool_use_id: str) -> str:
    return _det_id("tool", session_id, tool_use_id)


def _agent_id_from_cwd(cwd: str | None) -> str:
    """Derive the agent_id used by tj onboard --claude-code: claude-code-<basename>."""
    if not cwd:
        return "claude-code-unknown"
    name = Path(cwd).name.lower() or "unknown"
    return f"claude-code-{name}"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # CC uses ISO-8601 with trailing Z
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _provider_for_model(model: str) -> str:
    """Best-effort provider inference from a Claude Code model name."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o3") or model.startswith("o4"):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    return "anthropic"  # Claude Code always uses Anthropic at present


# --- Claude Code parser ------------------------------------------------------

def parse_claude_code_session(path: Path) -> ParsedSession | None:
    """
    Parse a single Claude Code JSONL session file.

    Returns None when the file contains no assistant turns (e.g. session
    ended before the first model call). Returns a ParsedSession with
    spans ready to be inserted.
    """
    session_id: str | None = None
    cwd: str | None = None
    earliest: datetime | None = None
    latest: datetime | None = None

    # Dedup by span_id WITHIN the session (#294). Claude Code replays/re-snapshots
    # assistant turns into the same JSONL on resume/branch — each appended record
    # gets a fresh `uuid` but the SAME `message.id` (the stable Anthropic API
    # response id) and same `requestId`. Keying span_id on message.id collapses
    # these to one span; `last-wins` keeps the finalized usage (early snapshots
    # carry partial output_tokens; the last record has the complete generation).
    spans_by_id: dict[str, NormalizedSpan] = {}

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(record, dict):
            continue

        if session_id is None:
            session_id = record.get("sessionId")
        if cwd is None:
            cwd = record.get("cwd")

        rtype = record.get("type")
        if rtype != "assistant":
            continue

        ts = _parse_ts(record.get("timestamp"))
        if ts is not None:
            if earliest is None or ts < earliest:
                earliest = ts
            if latest is None or ts > latest:
                latest = ts

        msg = record.get("message") or {}
        if not isinstance(msg, dict):
            continue

        model = msg.get("model")
        usage = msg.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens") or 0)

        # Some records have no model (e.g. early init); skip
        if not model:
            continue
        # Skip empty-usage records entirely (no cost contribution)
        if input_tokens == 0 and output_tokens == 0 and cache_read == 0 and cache_creation == 0:
            continue

        # Prefer the Anthropic API response id (`message.id`, msg_…) — it is
        # STABLE per real call and survives resume/branch re-ingest, where the
        # record `uuid` is regenerated (#294). Fall back to the record uuid only
        # when message.id is absent. NEVER key on the token-usage signature:
        # distinct real calls can share identical token counts.
        message_key = msg.get("id") or record.get("uuid") or f"{line_no}"
        sid_str = session_id or path.stem
        # One trace per session (#243): all assistant turns + their tool calls
        # in this conversation share a trace_id. span_id is keyed on the stable
        # message.id so idempotency holds across resumed/branched sessions.
        trace_id = _trace_id_for(sid_str)
        span_id = _span_id_for_assistant(sid_str, message_key)

        provider = _provider_for_model(model)
        cost = calculate_cost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_creation,
        )
        # Persist the cache read/write split, mirroring the live ingest path
        # (#245). cache_tokens = cache-READ only; cache_write_tokens =
        # cache-CREATION (priced higher). Collapsing them into one field made
        # the Cost table show CACHE W = 0 and a CACHE R that was actually
        # read+write. See models.py NormalizedSpan + Critical Rule on cache.

        agent_id = _agent_id_from_cwd(cwd)
        start_time = ts or datetime.now(tz=timezone.utc)
        # Duration unknown from on-disk format; leave None.
        # last-wins: a later replay of the same message.id overwrites earlier,
        # partial snapshots so the finalized usage/cost is the one we keep (#294).
        spans_by_id[span_id] = NormalizedSpan(
            span_id=span_id,
            trace_id=trace_id,
            name="gen_ai.llm.call",
            kind=SpanKind.CLIENT,
            status_code=SpanStatus.OK,
            start_time=start_time,
            end_time=start_time,
            duration_ms=None,
            agent_id=agent_id,
            session_id=sid_str,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_tokens=cache_read,
            cache_write_tokens=cache_creation,
            cost_usd=cost,
            request_type="completion",
            conversation_id=sid_str,
            attributes={"source": "backfill.claude_code"},
            billing_account="anthropic",
        )

        # Tool uses inside the assistant message become tool spans. tool_use `id`
        # is stable across resumes (verified in real data), so keying on it
        # collapses replays the same way. A per-message index (not a global
        # counter) keeps the no-id fallback deterministic across re-ingest.
        content = msg.get("content") or []
        if isinstance(content, list):
            for tool_idx, item in enumerate(b for b in content if isinstance(b, dict)
                                            and b.get("type") == "tool_use"):
                tool_use_id = item.get("id") or _det_id(
                    "tool-fallback", sid_str, message_key, str(tool_idx)
                )
                tool_span_id = _span_id_for_tool(sid_str, tool_use_id)
                tool_name = item.get("name") or "unknown"
                spans_by_id[tool_span_id] = NormalizedSpan(
                    span_id=tool_span_id,
                    trace_id=trace_id,
                    parent_span_id=span_id,
                    name="gen_ai.tool.call",
                    kind=SpanKind.INTERNAL,
                    status_code=SpanStatus.OK,
                    start_time=start_time,
                    end_time=start_time,
                    duration_ms=None,
                    agent_id=agent_id,
                    session_id=sid_str,
                    tool_name=tool_name,
                    conversation_id=sid_str,
                    attributes={"source": "backfill.claude_code"},
                )

    if not spans_by_id or session_id is None:
        return None

    # Totals are computed from the DEDUPED spans (#294) — never from per-record
    # accumulation, which would re-count every replayed snapshot. cache_tokens is
    # cache-READ only, matching the live path + SessionRecord semantics.
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

    agent_id = _agent_id_from_cwd(cwd)
    started_at = earliest or datetime.now(tz=timezone.utc)
    ended_at = latest or started_at

    return ParsedSession(
        session_id=session_id,
        agent_id=agent_id,
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


def iter_claude_code_sessions(
    root: Path | None = None,
    since: datetime | None = None,
) -> Iterator[ParsedSession]:
    """
    Walk a Claude Code projects directory and yield ParsedSession objects.

    `since` filters out files whose mtime is before the cutoff (cheap pre-filter);
    the actual session start_time is checked again per-file.
    """
    base = root or CLAUDE_CODE_PROJECTS_ROOT
    if not base.exists() or not base.is_dir():
        return
    for jsonl_path in sorted(base.rglob("*.jsonl")):
        try:
            if since is not None:
                mtime = datetime.fromtimestamp(jsonl_path.stat().st_mtime, tz=timezone.utc)
                if mtime < since:
                    continue
        except OSError:
            continue
        parsed = parse_claude_code_session(jsonl_path)
        if parsed is None:
            continue
        if since is not None and parsed.ended_at < since:
            continue
        yield parsed


def session_record_from_parsed(
    parsed: ParsedSession, plan_tier: str = "unknown",
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

def ingest_claude_code(
    db,
    root: Path | None = None,
    since: datetime | None = None,
    progress=None,
    config=None,
) -> BackfillResult:
    """
    Ingest Claude Code sessions into the storage backend.

    `db` is a DuckDBBackend (or compatible). Writes are idempotent: spans whose
    span_id already exists are skipped via INSERT … ON CONFLICT DO NOTHING.

    `config` (a TjConfig) supplies the declared plan tier so backfilled sessions
    carry the same `plan_tier` the live ingest path would set (#176). When None
    or no plan is configured, sessions fall back to "unknown" (prior behavior).

    `progress(parsed_session, result)` is called once per session if provided.
    """
    result = BackfillResult()
    projects_seen: set[str] = set()
    plan_tier = _plan_tier_for_provider(config, _CLAUDE_CODE_PROVIDER)
    for parsed in iter_claude_code_sessions(root=root, since=since):
        result.sessions_seen += 1
        result.seen_session_ids.add(parsed.session_id)
        if parsed.cwd:
            projects_seen.add(parsed.cwd)
        try:
            inserted = _insert_session_idempotent(db, parsed, plan_tier=plan_tier)
        except Exception as exc:
            result.files_failed += 1
            if len(result.sample_errors) < 5:
                result.sample_errors.append(f"{parsed.session_id}: {exc}")
            continue

        result.spans_ingested += inserted
        result.spans_skipped_existing += len(parsed.spans) - inserted
        if inserted > 0:
            result.sessions_ingested += 1
            result.new_session_ids.add(parsed.session_id)
        # Accumulate cost for every parsed file (not just files with new spans)
        # so the summary reports the full in-window total, not a new-only figure
        # that reads as "barely worked" on an idempotent re-run (#238).
        result.total_cost_usd += parsed.total_cost_usd

        if result.earliest is None or parsed.started_at < result.earliest:
            result.earliest = parsed.started_at
        if result.latest is None or parsed.ended_at > result.latest:
            result.latest = parsed.ended_at

        if progress is not None:
            try:
                progress(parsed, result)
            except Exception:
                pass

    # A Claude Code session is split across files that share one session_id
    # (main thread + subagents/agent-*.jsonl). The per-file upsert above uses
    # replace semantics, so each touched session row must be reconciled to the
    # SUM of its spans -- otherwise it holds only the last file's totals.
    # Idempotent: a re-run also repairs rows written by an earlier backfill.
    recompute = getattr(db, "recompute_session_totals_from_spans", None)
    if recompute is not None and result.seen_session_ids:
        recompute(sorted(result.seen_session_ids))
    result.project_count = len(projects_seen)
    return result


def _insert_session_idempotent(
    db, parsed: ParsedSession, plan_tier: str = "unknown",
) -> int:
    """
    Insert spans + session record; skip spans already present.
    Returns the number of newly-inserted spans.
    """
    conn = getattr(db, "conn", None)
    inserted = 0
    if conn is None:
        # Fall back to plain inserts when running against a backend that has no conn
        for span in parsed.spans:
            try:
                db.insert_span(span)
                inserted += 1
            except Exception:
                continue
        db.upsert_session(session_record_from_parsed(parsed, plan_tier))
        return inserted

    for span in parsed.spans:
        # PRIMARY KEY conflicts on (span_id) mean a previous backfill (or live ingest)
        # has already covered this span. Skip silently — the row count returned by
        # DuckDB's execute() isn't reliable for ON CONFLICT, so we pre-check.
        exists = conn.execute(
            "SELECT 1 FROM spans WHERE span_id = $1", [span.span_id]
        ).fetchone()
        if exists:
            continue
        conn.execute(
            "INSERT INTO spans ("
            "span_id, trace_id, parent_span_id, session_id, agent_id, "
            "name, kind, status_code, status_message, start_time, end_time, "
            "duration_ms, attributes, provider, model, tool_name, "
            "input_tokens, output_tokens, cache_tokens, cost_usd, "
            "request_type, conversation_id, events, billing_account, "
            "cache_write_tokens"
            ") VALUES "
            "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25)",
            [
                span.span_id, span.trace_id, span.parent_span_id, span.session_id,
                span.agent_id, span.name, span.kind.value, span.status_code.value,
                span.status_message, span.start_time, span.end_time, span.duration_ms,
                json.dumps(span.attributes), span.provider, span.model, span.tool_name,
                span.input_tokens, span.output_tokens, span.cache_tokens, span.cost_usd,
                span.request_type, span.conversation_id, json.dumps(span.events),
                span.billing_account, span.cache_write_tokens,
            ],
        )
        inserted += 1

    db.upsert_session(session_record_from_parsed(parsed, plan_tier))
    return inserted


__all__ = [
    "BackfillResult",
    "ParsedSession",
    "CLAUDE_CODE_PROJECTS_ROOT",
    "parse_claude_code_session",
    "iter_claude_code_sessions",
    "ingest_claude_code",
    "session_record_from_parsed",
]
