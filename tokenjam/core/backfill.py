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
from tokenjam.core.method_capture import capture_session_method
from tokenjam.core.models import (
    NormalizedSpan,
    SessionRecord,
    SpanKind,
    SpanStatus,
)
from tokenjam.core.transcript import _block_text
from tokenjam.core.usage import assistant_message_key, parse_usage
from tokenjam.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)


CLAUDE_CODE_PROJECTS_ROOT = Path.home() / ".claude" / "projects"

# The `attributes.source` tag every Claude Code backfill span carries (LLM +
# tool). Used to scope the stale-scheme reconciliation DELETE so it only ever
# touches backfill-sourced rows, never live-ingested spans.
_CLAUDE_CODE_SOURCE = "backfill.claude_code"

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
    spans_retagged: int = 0
    # Stale-scheme backfill spans purged this run (the #294/#300 cross-version
    # self-heal). 0 on a clean current-scheme DB; >0 the first time an affected
    # user re-backfills a DB that still holds pre-v0.5.2 uuid-keyed rows.
    spans_stale_purged: int = 0
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

    # True when a `max_sessions` cap was hit, so callers know more sessions exist
    # on disk than were ingested (the #13 quickstart first-run cap). The full
    # `tj backfill claude-code` path passes no cap, so this stays False there.
    limit_reached: bool = False


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
        # Four-bucket parse via the shared source of truth (core.usage), so the
        # statusline's re-read % and the Cost tab agree on the same session.
        usage = parse_usage(msg.get("usage"))
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read = usage.cache_read_tokens
        cache_creation = usage.cache_write_tokens

        # Some records have no model (e.g. early init); skip
        if not model:
            continue
        # Skip empty-usage records entirely (no cost contribution)
        if usage.total == 0:
            continue

        # Stable per-call dedup key (message.id, falling back to uuid/line_no);
        # keying span_id on it collapses resume/branch replays (#294). See
        # core.usage.assistant_message_key for the precedence + rationale.
        message_key = assistant_message_key(record, msg, line_no)
        sid_str = session_id or path.stem
        # One trace per session (#243): all assistant turns + their tool calls
        # in this conversation share a trace_id. span_id is keyed on the stable
        # message.id so idempotency holds across resumed/branched sessions.
        trace_id = _trace_id_for(sid_str)
        span_id = _span_id_for_assistant(sid_str, message_key)

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
        # Persist the cache read/write split, mirroring the live ingest path
        # (#245). cache_tokens = cache-READ only; cache_write_tokens =
        # cache-CREATION (priced higher). Collapsing them into one field made
        # the Cost table show CACHE W = 0 and a CACHE R that was actually
        # read+write. See models.py NormalizedSpan + Critical Rule on cache.

        agent_id = _agent_id_from_cwd(cwd)
        start_time = ts or datetime.now(tz=timezone.utc)

        # Per-message content (opt-in, gated by [capture]). Default-off leaves
        # llm_attrs == {"source": ...} so existing behavior is byte-for-byte
        # unchanged. Keys match GenAIAttributes so downstream consumers (and
        # alert content-stripping) treat backfilled content like live content.
        llm_attrs: dict = {"source": _CLAUDE_CODE_SOURCE}
        if capture.prompts and pending_prompt.strip():
            llm_attrs[GenAIAttributes.PROMPT_CONTENT] = pending_prompt
        if capture.completions:
            completion_text = _block_text(msg.get("content"))
            if completion_text.strip():
                llm_attrs[GenAIAttributes.COMPLETION_CONTENT] = completion_text
        # The prompt is consumed by exactly one assistant span.
        pending_prompt = ""

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
                tool_attrs: dict = {"source": _CLAUDE_CODE_SOURCE}
                if capture.tool_inputs:
                    tool_input = item.get("input")
                    # Persist whatever shape CC emitted (usually a dict);
                    # None/absent inputs add nothing.
                    if tool_input is not None:
                        tool_attrs[GenAIAttributes.TOOL_INPUT] = tool_input

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
                    sub_agent_id=sub_agent_id,
                    session_id=sid_str,
                    tool_name=tool_name,
                    conversation_id=sid_str,
                    attributes=tool_attrs,
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
    capture: CaptureConfig | None = None,
    max_sessions: int | None = None,
) -> Iterator[ParsedSession]:
    """
    Walk a Claude Code projects directory and yield ParsedSession objects.

    `since` filters out files whose mtime is before the cutoff (cheap pre-filter);
    the actual session start_time is checked again per-file.

    `capture` is forwarded to `parse_claude_code_session` to gate per-message
    content extraction (#3); None/all-False means no content is extracted.

    `max_sessions` caps how many sessions are parsed+yielded. When set, files are
    walked **most-recent first** (by mtime) and parsing stops once `max_sessions`
    sessions have been yielded — so the work this generator does (and the inserts
    its caller performs) is bounded regardless of how large `~/.claude` is. This
    powers the `tj quickstart` first-run cap (#13): a brand-new user with
    thousands of sessions sees the headline over their most-recent N sessions in
    bounded time, with the full picture available on demand. `None` (the default)
    keeps the original deterministic path-sorted, unbounded walk so the full
    `tj backfill claude-code` ingest is byte-for-byte unchanged.
    """
    base = root or CLAUDE_CODE_PROJECTS_ROOT
    if not base.exists() or not base.is_dir():
        return

    paths = list(base.rglob("*.jsonl"))
    if max_sessions is not None:
        # Most-recent first so the cap keeps the freshest sessions. We sort by
        # mtime (cheap, no parse) and read the stat once, reusing it for the
        # `since` pre-filter below.
        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:
                return 0.0

        paths = sorted(paths, key=_mtime, reverse=True)
    else:
        paths = sorted(paths)

    yielded = 0
    for jsonl_path in paths:
        if max_sessions is not None and yielded >= max_sessions:
            return
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
        yielded += 1


def count_claude_code_sessions_in_scope(
    root: Path | None = None,
    since: datetime | None = None,
    max_sessions: int | None = None,
) -> int:
    """Cheaply count how many Claude Code session files `ingest_claude_code`
    would walk for the given `root`/`since`/`max_sessions` — `stat()` calls
    only, no file is opened or parsed.

    Mirrors `iter_claude_code_sessions`'s file selection (the `since` mtime
    pre-filter, the `max_sessions` cap) closely enough to size a progress bar
    or print a heads-up before a potentially slow ingest starts (#443); it is
    NOT exact (it counts conversation *files*, matching `sessions_seen`, not
    the post-parse distinct-session count a full ingest reports), but it's
    the same cheap estimate `tj quickstart`'s first-run cap already accepts.
    """
    base = root or CLAUDE_CODE_PROJECTS_ROOT
    if not base.exists() or not base.is_dir():
        return 0
    paths = list(base.rglob("*.jsonl"))
    if since is not None:
        kept = []
        for p in paths:
            try:
                mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if mtime >= since:
                kept.append(p)
        paths = kept
    total = len(paths)
    return min(total, max_sessions) if max_sessions is not None else total


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

# Spans accumulated before a columnar bulk-append flush. Batching across sessions
# amortizes `read_json`'s per-call fixed cost and collapses the existence check
# into ONE set-based query per flush (vs one per session) — the win on a history
# of thousands of small sessions. A batch is a few MB of NDJSON, streamed to a
# temp file, so memory stays flat regardless of total history size.
_BULK_FLUSH_SPAN_TARGET = 25_000


def _record_insert_outcome(
    result: BackfillResult, parsed: ParsedSession, inserted: int, retagged: int
) -> None:
    """Fold one session's insert counts into the running `BackfillResult`."""
    result.spans_ingested += inserted
    result.spans_retagged += retagged
    result.spans_skipped_existing += len(parsed.spans) - inserted - retagged
    if inserted > 0:
        result.sessions_ingested += 1
        result.new_session_ids.add(parsed.session_id)


def _apply_session(
    db, parsed: ParsedSession, plan_tier: str, reingest: bool, result: BackfillResult
) -> None:
    """Per-session insert path (reingest, no-conn fallback, and the bulk-flush
    error fallback). Mirrors the historical per-file behavior including the
    `files_failed` / `sample_errors` accounting on a DB error."""
    try:
        inserted, retagged = _insert_session_idempotent(
            db, parsed, plan_tier=plan_tier, reingest=reingest
        )
    except Exception as exc:
        result.files_failed += 1
        if len(result.sample_errors) < 5:
            result.sample_errors.append(f"{parsed.session_id}: {exc}")
        return
    _record_insert_outcome(result, parsed, inserted, retagged)


def _bulk_apply_batch(
    db, batch: list[ParsedSession], plan_tier: str, result: BackfillResult
) -> None:
    """Insert an accumulated batch of sessions with ONE set-based existence
    check and ONE columnar bulk-append, then upsert each session row.

    Idempotency is IDENTICAL to the per-session path: a span already in the DB
    is skipped (existence check + the bulk-append anti-join), so a re-run inserts
    nothing. Session upserts stay per-file, in order, so `started_at` is set by
    the first-processed file exactly as before; `recompute_session_totals_from_spans`
    (run after the loop) reconciles token/cost totals from the spans regardless.
    """
    conn = db.conn
    # ONE existence check for the whole batch. span_ids are deterministic and
    # session-scoped so they're unique across distinct sessions, but a session
    # split across sibling files is yielded as separate ParsedSessions sharing a
    # session_id — guard intra-batch duplicates defensively so the bulk-append
    # never carries the same span_id twice.
    all_ids: list[str] = []
    seen: set[str] = set()
    for parsed in batch:
        for span in parsed.spans:
            if span.span_id not in seen:
                seen.add(span.span_id)
                all_ids.append(span.span_id)
    existing = _existing_span_ids(conn, all_ids)

    new_spans: list[NormalizedSpan] = []
    queued: set[str] = set()
    outcomes: list[tuple[ParsedSession, int]] = []
    for parsed in batch:
        inserted = 0
        for span in parsed.spans:
            if span.span_id in existing or span.span_id in queued:
                continue
            queued.add(span.span_id)
            new_spans.append(span)
            inserted += 1
        outcomes.append((parsed, inserted))

    try:
        db.bulk_insert_spans(new_spans)
    except Exception:
        # The INSERT..SELECT is atomic, so nothing landed — degrade to the
        # slow-but-safe per-session path (which also records files_failed) rather
        # than aborting the whole ingest. Re-running per session double-counts
        # nothing because no batch row was inserted.
        logger.warning(
            "bulk span append failed; falling back to per-session", exc_info=True
        )
        for parsed in batch:
            _apply_session(db, parsed, plan_tier, reingest=False, result=result)
        return

    for parsed, inserted in outcomes:
        db.upsert_session(session_record_from_parsed(parsed, plan_tier))
        _record_insert_outcome(result, parsed, inserted, 0)


def ingest_claude_code(
    db,
    root: Path | None = None,
    since: datetime | None = None,
    progress=None,
    config=None,
    reingest: bool = False,
    max_sessions: int | None = None,
) -> BackfillResult:
    """
    Ingest Claude Code sessions into the storage backend.

    `db` is a DuckDBBackend (or compatible). Writes are idempotent: spans whose
    span_id already exists are skipped (a batched existence check plus the
    columnar bulk-append's anti-join), so a re-run inserts no duplicates.

    `config` (a TjConfig) supplies the declared plan tier so backfilled sessions
    carry the same `plan_tier` the live ingest path would set (#176). When None
    or no plan is configured, sessions fall back to "unknown" (prior behavior).

    `config.capture` also gates per-message content extraction (#3): the same
    four `[capture]` toggles the live ingest path honors. Default-off (the
    config default, and when `config` is None) extracts no content, so a
    default backfill is byte-for-byte unchanged.

    `max_sessions` caps the number of (most-recent) sessions ingested so the
    work is bounded on a large `~/.claude` history — the #13 quickstart first-run
    cap. When the cap is hit, `result.limit_reached` is set True. `None` (the
    default, used by the full `tj backfill claude-code` path) ingests everything
    in window, unchanged.

    `progress(parsed_session, result)` is called once per session if provided.
    """
    result = BackfillResult()
    projects_seen: set[str] = set()
    plan_tier = _plan_tier_for_provider(config, _CLAUDE_CODE_PROVIDER)
    capture = getattr(config, "capture", None) if config is not None else None
    # Union of CURRENT-scheme (message.id-keyed) span_ids per session, aggregated
    # across ALL of a session's on-disk files (main thread +
    # subagents/agent-*.jsonl share one session_id). Used AFTER the loop for the
    # stale-scheme reconciliation DELETE — building the union first is essential:
    # a per-file DELETE scoped to session_id would wipe the sibling files' spans,
    # since they carry the same session_id + source tag (#294/#300).
    keep_by_session: dict[str, set[str]] = {}

    # A fresh full backfill (the ~8min/5.6GB hot path) accumulates spans across
    # sessions and flushes them through the columnar bulk-append: ONE set-based
    # existence check + ONE `read_json` vectorized insert per batch instead of a
    # dedup query + insert per session. `reingest` keeps the per-session path (its
    # per-span attribute overlay is not a bulk op); a backend without a `conn`
    # (defensive) also falls back per session.
    conn = getattr(db, "conn", None)
    use_bulk = conn is not None and not reingest
    batch: list[ParsedSession] = []
    batch_spans = 0

    def _flush_batch() -> None:
        nonlocal batch, batch_spans
        if batch:
            _bulk_apply_batch(db, batch, plan_tier, result)
            batch = []
            batch_spans = 0

    for parsed in iter_claude_code_sessions(
        root=root, since=since, capture=capture, max_sessions=max_sessions,
    ):
        result.sessions_seen += 1
        result.seen_session_ids.add(parsed.session_id)
        keep_by_session.setdefault(parsed.session_id, set()).update(
            s.span_id for s in parsed.spans
        )
        if parsed.cwd:
            projects_seen.add(parsed.cwd)

        # Cost + window bounds are per-file totals, independent of whether the
        # spans are new. Accumulate for every parsed file so the summary reports
        # the full in-window total, not a new-only figure that reads as "barely
        # worked" on an idempotent re-run (#238).
        result.total_cost_usd += parsed.total_cost_usd
        if result.earliest is None or parsed.started_at < result.earliest:
            result.earliest = parsed.started_at
        if result.latest is None or parsed.ended_at > result.latest:
            result.latest = parsed.ended_at

        if use_bulk:
            batch.append(parsed)
            batch_spans += len(parsed.spans)
            if batch_spans >= _BULK_FLUSH_SPAN_TARGET:
                _flush_batch()
        else:
            _apply_session(db, parsed, plan_tier, reingest, result)

        if progress is not None:
            try:
                progress(parsed, result)
            except Exception:
                pass

    _flush_batch()

    # Self-heal stale-scheme duplicates (#294/#300 cross-version). A DB written
    # by <=v0.5.1 keyed backfill span_ids on the record `uuid`; current code keys
    # on the stable `message.id`. The two schemes are DISJOINT, so re-backfilling
    # an old DB ADDS a full duplicate set alongside the stale rows, inflating
    # token/cost totals ~2.6x. `keep_by_session[sid]` is the COMPLETE
    # current-scheme span_id set for the session (LLM + tool spans, unioned across
    # all its files); any `backfill.claude_code`-tagged span for that session NOT
    # in the set can only be a stale-scheme orphan, so drop it. Scoped to
    # (session_id, source) -> never touches live-ingested spans or other sessions.
    # Runs BEFORE recompute so the reconciled sums exclude the purged rows.
    # Skipped under the `max_sessions` quickstart cap: that path stops mid-session
    # (bounded preview), so its per-session keep-set may be incomplete and a
    # DELETE could drop valid spans; the full `tj backfill claude-code` path
    # (used by onboard) does the self-healing.
    reconcile = getattr(db, "reconcile_backfill_spans", None)
    if reconcile is not None and max_sessions is None and keep_by_session:
        try:
            purged = reconcile(keep_by_session, _CLAUDE_CODE_SOURCE)
            result.spans_stale_purged = purged
        except Exception as exc:  # never let reconciliation break the ingest
            logger.warning("stale-span reconciliation skipped: %s", exc)

    # A Claude Code session is split across files that share one session_id
    # (main thread + subagents/agent-*.jsonl). The per-file upsert above uses
    # replace semantics, so each touched session row must be reconciled to the
    # SUM of its spans -- otherwise it holds only the last file's totals.
    # Idempotent: a re-run also repairs rows written by an earlier backfill.
    recompute = getattr(db, "recompute_session_totals_from_spans", None)
    if recompute is not None and result.seen_session_ids:
        recompute(sorted(result.seen_session_ids))

    # Snapshot each newly-ingested session's reconstructed method into
    # `session_story` so it survives Claude Code pruning the transcript later
    # (the whole point of the persistence path — historical sessions are the
    # ones most likely to lose their on-disk file). We capture ONLY the sessions
    # that gained new spans this run (`new_session_ids`); an idempotent re-run
    # re-captures nothing. Cost: one extra Story build per ingested session,
    # re-reading the transcript backfill just parsed off `root`. Best-effort —
    # capture_session_method swallows its own errors and never raises, so it
    # cannot change backfill's result or break the ingest.
    for sid in sorted(result.new_session_ids):
        capture_session_method(db, sid, projects_dir=root, source="backfill")

    result.project_count = len(projects_seen)
    # The iterator stops yielding once the cap is reached, so seeing exactly
    # `max_sessions` means there may be older sessions on disk we skipped.
    if max_sessions is not None and result.sessions_seen >= max_sessions:
        result.limit_reached = True
    return result


def _load_attrs(conn, span_id: str) -> dict:
    """Read a stored span's `attributes` column as a dict.

    DuckDB may hand the JSON column back as a string or an already-parsed
    object depending on backend; normalize both to a dict. Malformed/missing
    attributes degrade to an empty dict so a reingest never raises.
    """
    row = conn.execute(
        "SELECT attributes FROM spans WHERE span_id = $1", [span_id]
    ).fetchone()
    if not row or row[0] is None:
        return {}
    value = row[0]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _merge_attributes(exists_attributes: dict, parsed_attributes: dict) -> dict:
    """Overlay freshly-parsed attributes over the stored ones (parsed wins
    per key), returning a NEW dict (the stored row is never mutated in place).

    This is the #10 backfill: when `[capture]` is enabled after a span was
    first ingested, the parsed span now carries content keys
    (`gen_ai.prompt.content` etc.) the stored row lacks. Merging ADDS those
    without dropping any keys the stored row already had. Capture-off reingest
    is a no-op because the parsed attributes are just `{"source": ...}`.
    """
    return {**exists_attributes, **parsed_attributes}


def _existing_span_ids(conn, span_ids: list[str]) -> set[str]:
    """Return the subset of `span_ids` that already exist in `spans`, in ONE
    query (replacing the per-span existence SELECT). Chunked so an enormous
    session can't blow past DuckDB's bind-parameter ceiling.
    """
    found: set[str] = set()
    if not span_ids:
        return found
    chunk = 5000
    for start in range(0, len(span_ids), chunk):
        batch = span_ids[start:start + chunk]
        placeholders = ",".join(f"${i + 1}" for i in range(len(batch)))
        rows = conn.execute(
            f"SELECT span_id FROM spans WHERE span_id IN ({placeholders})", batch
        ).fetchall()
        found.update(r[0] for r in rows)
    return found


def _insert_session_idempotent(
    db, parsed: ParsedSession, plan_tier: str = "unknown", reingest: bool = False
) -> tuple[int, int]:
    """
    Insert spans + session record; skip spans already present.
    Returns (newly_inserted, retagged).

    Per-session path: ONE (chunked) `WHERE span_id IN (...)` partitions spans
    into new-vs-existing, then the new spans are appended via the columnar
    `db.bulk_insert_spans` (newline-delimited JSON + DuckDB `read_json`, ~350×
    faster than per-row binding). Its anti-join skips any span_id already present
    (a previous backfill or live ingest already covered it). The full backfill
    (`reingest=False`) drives the even faster cross-session batch path in
    `ingest_claude_code`; this per-session routine remains the `reingest=True`
    path (its per-span attribute overlay is not a bulk op) and the fallback for a
    backend without a `conn`.

    When `reingest` is True, spans that already exist are UPDATEd instead of
    being skipped — this backfills two things onto rows an older/leaner backfill
    wrote:
      - `sub_agent_id` — re-tags history ingested before that column existed.
      - `attributes`   — overlays freshly-parsed captured content
        (`gen_ai.prompt.content` / `gen_ai.completion.content` /
        `gen_ai.tool.input`) onto the stored row when `[capture]` was enabled
        AFTER the span was first ingested (#10). Without this, enabling capture
        later never lands content on already-ingested spans, so the
        recurring-inclusion detection #4 needs (which reads that content) only
        worked against a fresh DB.

    The overlay is a per-key merge of the parsed span's attributes over the
    stored attributes (parsed wins per key) — so it ADDS content keys without
    discarding any keys the stored row already carried (e.g. from live ingest).
    Capture-off reingest is a no-op: the parsed span's attributes are just
    `{"source": ...}`, which the stored row already has, so nothing changes.
    Other span fields are left untouched.
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

    span_ids = [s.span_id for s in parsed.spans]
    existing = _existing_span_ids(conn, span_ids)
    new_spans = [s for s in parsed.spans if s.span_id not in existing]
    if new_spans:
        db.bulk_insert_spans(new_spans)
        inserted = len(new_spans)

    if reingest and existing:
        # Re-tag rows an older/leaner backfill wrote: overlay sub_agent_id (for
        # history ingested before that column existed) and any freshly-parsed
        # captured content (#10) so enabling [capture] later backfills onto
        # already-ingested spans. Per-span UPDATE, bounded by the existing set.
        for span in parsed.spans:
            if span.span_id not in existing:
                continue
            merged_attrs = _merge_attributes(
                exists_attributes=_load_attrs(conn, span.span_id),
                parsed_attributes=span.attributes,
            )
            conn.execute(
                "UPDATE spans SET sub_agent_id = $1, attributes = $2 "
                "WHERE span_id = $3",
                [span.sub_agent_id, json.dumps(merged_attrs), span.span_id],
            )
            retagged += 1

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
    "count_claude_code_sessions_in_scope",
]
