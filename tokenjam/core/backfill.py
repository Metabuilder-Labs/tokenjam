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
from tokenjam.core.config import CaptureConfig
from tokenjam.core.models import (
    NormalizedSpan,
    SessionRecord,
    SpanKind,
    SpanStatus,
)
from tokenjam.core.transcript import _block_text
from tokenjam.otel.semconv import GenAIAttributes

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
    sessions_seen: int = 0
    sessions_ingested: int = 0
    spans_ingested: int = 0
    spans_skipped_existing: int = 0
    spans_retagged: int = 0
    files_failed: int = 0
    earliest: datetime | None = None
    latest: datetime | None = None
    total_cost_usd: float = 0.0
    project_count: int = 0
    sample_errors: list[str] = field(default_factory=list)


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
    total_cache_write_tokens: int
    total_cost_usd: float
    tool_call_count: int


# --- ID derivation helpers ---------------------------------------------------

def _det_id(*parts: str, length: int = 16) -> str:
    """Deterministic hex ID derived from the given parts."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return h[:length]


def _trace_id_for(session_id: str, message_uuid: str) -> str:
    return _det_id("trace", session_id, message_uuid, length=32)


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


def _user_prompt_text(record: dict) -> str:
    """Extract the human prompt text from a Claude Code ``user`` record.

    A ``user`` record's ``message.content`` is either a plain string (the
    prompt) or a list of ``tool_result`` blocks (a tool turn, no prompt). We
    surface only the former — ``_block_text`` returns the string as-is and the
    empty string for a tool-result-only turn. Used to attach the triggering
    prompt to the next assistant span when ``capture.prompts`` is on.
    """
    msg = record.get("message")
    if not isinstance(msg, dict):
        return ""
    return _block_text(msg.get("content"))


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

def parse_claude_code_session(
    path: Path, capture: CaptureConfig | None = None,
) -> ParsedSession | None:
    """
    Parse a single Claude Code JSONL session file.

    Returns None when the file contains no assistant turns (e.g. session
    ended before the first model call). Returns a ParsedSession with
    spans ready to be inserted.

    ``capture`` gates per-message content extraction (#3), honoring the same
    four ``[capture]`` toggles the live ingest path enforces via
    ``strip_captured_content``. Default ``None`` (and the all-False default)
    leaves every span's ``attributes`` exactly as before — content extraction
    is strictly opt-in and stays 100% local:
      - ``capture.prompts``     -> ``gen_ai.prompt.content`` (the triggering
                                   human prompt) on the assistant LLM span.
      - ``capture.completions`` -> ``gen_ai.completion.content`` (the agent's
                                   narration text) on the assistant LLM span.
      - ``capture.tool_inputs`` -> ``gen_ai.tool.input`` (the raw tool args)
                                   on each tool span.
    The transcript carries no per-call tool *output*, so ``capture.tool_outputs``
    has nothing to extract on the backfill path.
    """
    capture = capture or CaptureConfig()
    # The user prompt that triggered the next assistant turn; reset after it is
    # consumed so a prompt is attributed to exactly one assistant span.
    pending_prompt: str = ""
    session_id: str | None = None
    cwd: str | None = None
    earliest: datetime | None = None
    latest: datetime | None = None

    spans: list[NormalizedSpan] = []
    total_input = 0
    total_output = 0
    total_cache = 0
    total_cache_write = 0
    total_cost = 0.0
    tool_count = 0

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
        if rtype == "user" and capture.prompts and not record.get("isMeta"):
            # Remember the latest genuine human prompt so the next assistant
            # span can carry it. Tool-result-only user turns yield "" and are
            # ignored (no prompt to attribute).
            prompt_text = _user_prompt_text(record)
            if prompt_text.strip():
                pending_prompt = prompt_text
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

        message_uuid = record.get("uuid") or msg.get("id") or f"{line_no}"
        sid_str = session_id or path.stem
        trace_id = _trace_id_for(sid_str, message_uuid)
        span_id = _span_id_for_assistant(sid_str, message_uuid)

        # Subagent attribution: Claude Code marks Task-tool (sidechain) turns
        # with a top-level `isSidechain` flag plus the subagent's own `agentId`.
        # Records in <session>/subagents/agent-<id>.jsonl carry these; main-thread
        # records don't. Stamp every span from this turn so a session's cost can
        # be broken down per subagent. None on the main thread.
        sub_agent_id = record.get("agentId") if record.get("isSidechain") else None

        provider = _provider_for_model(model)
        cost = calculate_cost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_creation,
        )

        agent_id = _agent_id_from_cwd(cwd)
        start_time = ts or datetime.now(tz=timezone.utc)

        # Per-message content (opt-in, gated by [capture]). Default-off leaves
        # llm_attrs == {"source": ...} so existing behavior is byte-for-byte
        # unchanged. Keys match GenAIAttributes so downstream consumers (and
        # alert content-stripping) treat backfilled content like live content.
        llm_attrs: dict = {"source": "backfill.claude_code"}
        if capture.prompts and pending_prompt.strip():
            llm_attrs[GenAIAttributes.PROMPT_CONTENT] = pending_prompt
        if capture.completions:
            completion_text = _block_text(msg.get("content"))
            if completion_text.strip():
                llm_attrs[GenAIAttributes.COMPLETION_CONTENT] = completion_text
        # The prompt is consumed by exactly one assistant span.
        pending_prompt = ""

        # Duration unknown from on-disk format; leave None
        spans.append(
            NormalizedSpan(
                span_id=span_id,
                trace_id=trace_id,
                name="gen_ai.llm.call",
                kind=SpanKind.CLIENT,
                status_code=SpanStatus.OK,
                start_time=start_time,
                end_time=start_time,
                duration_ms=None,
                agent_id=agent_id,
                sub_agent_id=sub_agent_id,
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
                attributes=llm_attrs,
                billing_account="anthropic",
            )
        )
        total_input += input_tokens
        total_output += output_tokens
        total_cache += cache_read
        total_cache_write += cache_creation
        total_cost += cost

        # Tool uses inside the assistant message become tool spans
        content = msg.get("content") or []
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "tool_use":
                    continue
                tool_use_id = item.get("id") or _det_id(
                    "tool-fallback", sid_str, message_uuid, str(tool_count)
                )
                tool_span_id = _span_id_for_tool(sid_str, tool_use_id)
                tool_name = item.get("name") or "unknown"

                tool_attrs: dict = {"source": "backfill.claude_code"}
                if capture.tool_inputs:
                    tool_input = item.get("input")
                    # Persist whatever shape CC emitted (usually a dict);
                    # None/absent inputs add nothing.
                    if tool_input is not None:
                        tool_attrs[GenAIAttributes.TOOL_INPUT] = tool_input

                spans.append(
                    NormalizedSpan(
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
                        sub_agent_id=sub_agent_id,
                        session_id=sid_str,
                        tool_name=tool_name,
                        conversation_id=sid_str,
                        attributes=tool_attrs,
                    )
                )
                tool_count += 1

    if not spans or session_id is None:
        return None

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
        total_cache_write_tokens=total_cache_write,
        total_cost_usd=round(total_cost, 8),
        tool_call_count=tool_count,
    )


def iter_claude_code_sessions(
    root: Path | None = None,
    since: datetime | None = None,
    capture: CaptureConfig | None = None,
) -> Iterator[ParsedSession]:
    """
    Walk a Claude Code projects directory and yield ParsedSession objects.

    `since` filters out files whose mtime is before the cutoff (cheap pre-filter);
    the actual session start_time is checked again per-file.

    `capture` is forwarded to `parse_claude_code_session` to gate per-message
    content extraction (#3); None/all-False means no content is extracted.
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
        parsed = parse_claude_code_session(jsonl_path, capture=capture)
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
        cache_write_tokens=parsed.total_cache_write_tokens,
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
    reingest: bool = False,
    config=None,
) -> BackfillResult:
    """
    Ingest Claude Code sessions into the storage backend.

    `db` is a DuckDBBackend (or compatible). Writes are idempotent: spans whose
    span_id already exists are skipped via INSERT … ON CONFLICT DO NOTHING.

    `config` (a TjConfig) supplies the declared plan tier so backfilled sessions
    carry the same `plan_tier` the live ingest path would set (#176). When None
    or no plan is configured, sessions fall back to "unknown" (prior behavior).

    `config.capture` also gates per-message content extraction (#3): the same
    four `[capture]` toggles the live ingest path honors. Default-off (the
    config default, and when `config` is None) extracts no content, so a
    default backfill is byte-for-byte unchanged.

    `progress(parsed_session, result)` is called once per session if provided.
    """
    result = BackfillResult()
    projects_seen: set[str] = set()
    seen_session_ids: set[str] = set()
    plan_tier = _plan_tier_for_provider(config, _CLAUDE_CODE_PROVIDER)
    capture = getattr(config, "capture", None) if config is not None else None
    for parsed in iter_claude_code_sessions(root=root, since=since, capture=capture):
        result.sessions_seen += 1
        if parsed.cwd:
            projects_seen.add(parsed.cwd)
        try:
            inserted, retagged = _insert_session_idempotent(
                db, parsed, reingest=reingest, plan_tier=plan_tier
            )
        except Exception as exc:
            result.files_failed += 1
            if len(result.sample_errors) < 5:
                result.sample_errors.append(f"{parsed.session_id}: {exc}")
            continue

        result.spans_ingested += inserted
        result.spans_retagged += retagged
        result.spans_skipped_existing += len(parsed.spans) - inserted - retagged
        seen_session_ids.add(parsed.session_id)
        if inserted > 0:
            result.sessions_ingested += 1
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
    if recompute is not None and seen_session_ids:
        recompute(sorted(seen_session_ids))
    result.project_count = len(projects_seen)
    return result


def _insert_session_idempotent(
    db, parsed: ParsedSession, reingest: bool = False, plan_tier: str = "unknown",
) -> tuple[int, int]:
    """
    Insert spans + session record; skip spans already present.
    Returns (newly_inserted, retagged).

    When `reingest` is True, spans that already exist are UPDATEd to refresh
    `sub_agent_id` instead of being skipped — this re-tags history ingested by
    an older backfill that predates the column. Other span fields are left
    untouched.
    """
    conn = getattr(db, "conn", None)
    inserted = 0
    retagged = 0
    if conn is None:
        # Fall back to plain inserts when running against a backend that has no conn
        for span in parsed.spans:
            try:
                db.insert_span(span)
                inserted += 1
            except Exception:
                continue
        db.upsert_session(session_record_from_parsed(parsed, plan_tier))
        return inserted, retagged

    for span in parsed.spans:
        # PRIMARY KEY conflicts on (span_id) mean a previous backfill (or live ingest)
        # has already covered this span. Skip silently — the row count returned by
        # DuckDB's execute() isn't reliable for ON CONFLICT, so we pre-check.
        exists = conn.execute(
            "SELECT 1 FROM spans WHERE span_id = $1", [span.span_id]
        ).fetchone()
        if exists:
            if reingest:
                # Re-tag history from before the sub_agent_id column existed.
                conn.execute(
                    "UPDATE spans SET sub_agent_id = $1 WHERE span_id = $2",
                    [span.sub_agent_id, span.span_id],
                )
                retagged += 1
            continue
        conn.execute(
            "INSERT INTO spans ("
            "span_id, trace_id, parent_span_id, session_id, agent_id, "
            "name, kind, status_code, status_message, start_time, end_time, "
            "duration_ms, attributes, provider, model, tool_name, "
            "input_tokens, output_tokens, cache_tokens, cost_usd, "
            "request_type, conversation_id, events, billing_account, "
            "cache_write_tokens, sub_agent_id"
            ") VALUES "
            "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26)",
            [
                span.span_id, span.trace_id, span.parent_span_id, span.session_id,
                span.agent_id, span.name, span.kind.value, span.status_code.value,
                span.status_message, span.start_time, span.end_time, span.duration_ms,
                json.dumps(span.attributes), span.provider, span.model, span.tool_name,
                span.input_tokens, span.output_tokens, span.cache_tokens, span.cost_usd,
                span.request_type, span.conversation_id, json.dumps(span.events),
                span.billing_account, span.cache_write_tokens, span.sub_agent_id,
            ],
        )
        inserted += 1

    db.upsert_session(session_record_from_parsed(parsed, plan_tier))
    return inserted, retagged


__all__ = [
    "BackfillResult",
    "ParsedSession",
    "CLAUDE_CODE_PROJECTS_ROOT",
    "parse_claude_code_session",
    "iter_claude_code_sessions",
    "ingest_claude_code",
    "session_record_from_parsed",
]
