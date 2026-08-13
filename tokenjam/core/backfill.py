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
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tokenjam.core.cost import calculate_cost
from tokenjam.core.distill import is_tokenjam_invoke_cwd
from tokenjam.core.pricing import classify_pricing_source
from tokenjam.core import system_prefix
from tokenjam.core.config import CaptureConfig
from tokenjam.core.method_capture import capture_session_method
from tokenjam.core.optimize.repeat_task import hash_task_statement
from tokenjam.core.models import (
    SESSION_STALE_THRESHOLD,
    NormalizedSpan,
    SessionRecord,
    SpanKind,
    SpanStatus,
)
from tokenjam.core.transcript import _block_text
from tokenjam.core.usage import assistant_message_key, parse_usage
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes

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
    # Transcript records declined for carrying no parseable timestamp — see
    # ParsedSession.records_undated. Surfaced by `tj backfill`'s summary so a
    # decline is never silent.
    records_undated: int = 0
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
    # On-disk mtime of the transcript file this was parsed from (tz-aware UTC),
    # or None when it couldn't be stat'd or the session was built directly (not
    # parsed from a file, e.g. in tests). Lets `session_record_from_parsed`
    # tell a transcript that is STILL being appended to apart from one that has
    # gone quiet, without re-deriving a second definition of "stale" (backfill
    # used to hardcode every session's status "completed" regardless).
    transcript_mtime: datetime | None = None
    # Records declined for carrying no parseable timestamp. Counted rather than
    # dropped in silence: a record tj refuses to ingest is a change to what the
    # corpus contains, and has to be as visible as one it accepts.
    records_undated: int = 0
    # The FIRST genuine human prompt seen in this file — captured
    # UNCONDITIONALLY (unlike `pending_prompt`'s per-span attachment above,
    # which is gated on `[capture] prompts`). This is never stored raw: the
    # only use is `session_record_from_parsed` hashing it into
    # `SessionRecord.task_statement_hash` via `repeat_task.hash_task_statement`
    # — a one-way, normalized fingerprint, not readable content, so it needs
    # no capture-toggle gate. None when the file has no main-thread user turn
    # (a subagent-only file, or a session with no recognizable prompt).
    first_user_prompt: str | None = None
    # The model that ran the most (input+output) tokens on the MAIN THREAD of
    # this file (subagent-dispatch spans excluded — a rightsized-down subagent
    # model shouldn't drown out what the session itself mostly ran on). None
    # when the file carries no priced assistant turn.
    dominant_model: str | None = None


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


#: ``taskKind`` values whose ``agentType`` is a caller-chosen, per-dispatch
#: INSTANCE LABEL rather than a reusable agent-definition name.
#:
#: THE EMPTY RESULT HERE IS DELIBERATE — do not "fix" it. An
#: ``in_process_teammate`` is spawned with an ad-hoc ``name`` ("worker-428",
#: "fix-499") and Claude Code writes that name into ``agentType``, so the field
#: is populated and looks perfectly usable. It is not: the label is minted per
#: dispatch, so adopting it would reintroduce exactly the unclusterable
#: one-session-per-identity property that ``sub_agent_type`` exists to remove,
#: and it addresses no definition file. Leaving these spans at
#: ``sub_agent_type = None`` reads like a gap in the extraction; recording them
#: would instead be a silent mis-attribution, which is worse than a known gap.
_PER_DISPATCH_TASK_KINDS = frozenset({"in_process_teammate"})


def _subagent_type_for(path: Path) -> str | None:
    """The dispatched agent TYPE for a Claude Code subagent transcript, or None.

    SOURCE, AND WHY NOT THE OBVIOUS ONE. Claude Code writes an
    ``agent-<agentId>.meta.json`` sidecar next to every subagent transcript,
    carrying ``agentType`` — the ``subagent_type`` argument of the spawning
    Task/Agent call, already resolved (a dispatch that omitted the argument
    reads as the default it actually ran under, rather than as absent).

    The natural-looking alternative is to join a sidechain to its dispatching
    ``Task`` call and read ``subagent_type`` off the tool args — a real field,
    which the next reader will find and assume is the source. Rejected because
    that join is only as available as the PARENT transcript, and Claude Code
    prunes transcripts on its own retention setting: wherever the parent has
    aged out, the dispatch is unlinkable, and subagent transcripts outlive their
    parents often enough that this is the common case rather than the edge. The
    sidecar sits beside the child and shares its lifetime, so it is present
    whenever the child is. The two were cross-checked against each other on real
    transcripts and they agree; where they differ it is because the Task call
    omitted the argument and the sidecar records the resolved default, so the
    sidecar is the more correct of the pair, not merely the more available.

    Sidechain records live ONLY under a ``subagents/`` directory, in
    ``agent-<id>.jsonl`` files, one agentId per file (checked both ways on real
    transcripts: no main-thread file carries an ``isSidechain`` record, and no
    subagent file carries more than one distinct ``agentId``), so one per-file
    lookup covers every span this file produces.

    The directory is NOT always the immediate parent: a workflow dispatch nests
    one level further, as ``subagents/workflows/<workflow-id>/agent-<id>.jsonl``.
    Membership is therefore tested against the whole path. A predicate matching
    only ``path.parent`` silently drops every workflow dispatch — silently
    because a missing type is indistinguishable from a dispatch that legitimately
    has none, so nothing raises and no count looks wrong locally. It surfaced
    only as a disagreement between the number of typed spans in the DB and the
    number of typed transcripts on disk.

    Returns None for a main-thread transcript, a missing/garbled sidecar, and a
    per-dispatch instance label (see ``_PER_DISPATCH_TASK_KINDS``).
    """
    if "subagents" not in path.parts or not path.name.startswith("agent-"):
        return None
    meta_path = path.with_suffix(".meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get("taskKind") in _PER_DISPATCH_TASK_KINDS:
        return None
    agent_type = meta.get("agentType")
    if not isinstance(agent_type, str) or not agent_type.strip():
        return None
    return agent_type.strip()


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


def _read_project_claude_md(cwd: str | None) -> str:
    """Best-effort read of `<cwd>/CLAUDE.md` — the one piece of Claude Code's
    actual system prompt that's recoverable on this machine (#272). The
    on-disk session transcript never records the system block CC sends the
    API (verified: it appears nowhere in a real transcript, even the first
    turn), so there is no captured field to source it from; CLAUDE.md is read
    straight off disk instead. It's genuinely resent unchanged on every call
    for this project — the chat-completions API is stateless, so the full
    system prompt goes out with every request — which is exactly the
    stable, repeated prefix cache-recommend looks for. Returns "" when cwd is
    unknown or the file is missing/unreadable; never raises.

    The text is read in full but **never stored** — see
    `_system_prefix_attrs`, which reduces it to the three values the analyzer
    actually consumes before it reaches a span.
    """
    if not cwd:
        return ""
    try:
        return (Path(cwd) / "CLAUDE.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _system_prefix_attrs(text: str) -> dict[str, Any]:
    """The prefix as it goes onto a span: identity, a display sample, a length.

    Storing the text itself is what took a real database to 4.06 GB — the value
    is identical across every span of a project, so its cost is (size x span
    count), and 92,514 spans were each holding a ~43 KB copy of one of 61
    distinct files. `cache_recommend` never used the text as text: it hashed a
    fixed-size head of it, kept 120 characters for display, and compared its
    length against a floor. All three survive here at ~230 bytes per span.

    Derivation lives in `core.system_prefix` so the producer and the consumer
    cannot drift on what a prefix's identity is.
    """
    summary = system_prefix.summarize(text)
    if not summary:
        return {}
    return {
        TjAttributes.SYSTEM_PREFIX_HASH: summary["hash"],
        TjAttributes.SYSTEM_PREFIX_SAMPLE: summary["sample"],
        TjAttributes.SYSTEM_PREFIX_LENGTH: summary["length"],
    }


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
                                   human prompt) on the assistant LLM span,
                                   plus ``tokenjam.system_prefix.content`` (the
                                   project's ``CLAUDE.md``, read straight off
                                   disk — the actually-reused cacheable prefix
                                   CC resends every call; #272) when found.
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
    records_undated: int = 0
    # Captured unconditionally (never gated on `[capture] prompts` — see
    # `ParsedSession.first_user_prompt`'s docstring for why that's safe).
    first_user_prompt: str | None = None
    # Main-thread-only per-model token totals, for `ParsedSession.dominant_model`.
    _model_tokens: dict[str, int] = {}

    # Dedup by span_id WITHIN the session (#294). Claude Code replays/re-snapshots
    # assistant turns into the same JSONL on resume/branch — each appended record
    # gets a fresh `uuid` but the SAME `message.id` (the stable Anthropic API
    # response id) and same `requestId`. Keying span_id on message.id collapses
    # these to one span; `last-wins` keeps the finalized usage (early snapshots
    # carry partial output_tokens; the last record has the complete generation).
    spans_by_id: dict[str, NormalizedSpan] = {}

    # Lazy-loaded once cwd is known (#272): the project's CLAUDE.md content,
    # captured as `TjAttributes.SYSTEM_PREFIX_CONTENT` below. `None` = not yet
    # attempted; `""` is a legitimate resolved outcome (no CLAUDE.md found).
    claude_md_text: str | None = None

    # The STABLE subagent identity for this file, resolved once (it is a
    # property of the transcript, not of a record — a subagents/agent-<id>.jsonl
    # holds exactly one agentId). None for a main-thread transcript.
    sub_agent_type = _subagent_type_for(path)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None

    # Stat once, alongside the read above, so `session_record_from_parsed` can
    # tell "still being appended to" apart from "gone quiet" without a second
    # filesystem round-trip. None when the file can't be stat'd (rare: it just
    # succeeded a read); a missing mtime degrades to the terminal ("completed")
    # side of that decision rather than raising.
    try:
        transcript_mtime: datetime | None = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        )
    except OSError:
        transcript_mtime = None

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
            # tokenjam's own distill/presence model calls (core.distill,
            # core.rulewrite.presence) shell out to this SAME `claude` CLI
            # from a private temp-dir cwd, and leave an ordinary transcript
            # on disk like any user session — never a session the user
            # actually worked. Excluded at the earliest possible point (the
            # first record that reveals cwd) rather than parsed and priced.
            if cwd is not None and is_tokenjam_invoke_cwd(cwd):
                return None

        rtype = record.get("type")
        if rtype == "user" and not record.get("isMeta"):
            # First genuine MAIN-THREAD human prompt, captured UNCONDITIONALLY
            # (never gated on `capture.prompts` — see `ParsedSession
            # .first_user_prompt`'s docstring for why that's safe: only ever
            # hashed, never stored raw). A subagent file's "user" turns are
            # the Task tool's dispatched instructions, not what the human
            # actually typed, so those are excluded here.
            if (
                first_user_prompt is None
                and not record.get("isSidechain")
                and _user_prompt_text(record).strip()
            ):
                first_user_prompt = _user_prompt_text(record)
            if capture.prompts:
                # Remember the latest genuine human prompt so the next
                # assistant span can carry it. Tool-result-only user turns
                # yield "" and are ignored (no prompt to attribute).
                prompt_text = _user_prompt_text(record)
                if prompt_text.strip():
                    pending_prompt = prompt_text
        if rtype != "assistant":
            continue

        ts = _parse_ts(record.get("timestamp"))
        if ts is None:
            # No observed time, so the record is not ingested rather than
            # ingested with a made-up one. `now` would date months-old
            # transcript work to whenever the backfill happened to run — which
            # reads as a real observation, and is why this is worse than the
            # 1970 sentinel it replaced rather than better. Same idiom as
            # ingest_adapters/langfuse.py and helicone.py.
            records_undated += 1
            continue
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
        # `sub_agent_id` stays PER-DISPATCH (analyzers group on
        # `(session_id, sub_agent_id)` to separate concurrent dispatches);
        # `sub_agent_type` is the stable, cross-session identity that names an
        # agent definition. Both are gated on the same isSidechain flag so a
        # main-thread span can never pick up a type.
        is_sidechain = bool(record.get("isSidechain"))
        sub_agent_id = record.get("agentId") if is_sidechain else None
        span_sub_agent_type = sub_agent_type if is_sidechain else None

        # Main-thread-only per-model token tally, for `ParsedSession
        # .dominant_model` — a rightsized-down subagent model shouldn't drown
        # out what the session itself mostly ran on.
        if not is_sidechain:
            _model_tokens[model] = _model_tokens.get(model, 0) + input_tokens + output_tokens

        provider = _provider_for_model(model)
        cost = calculate_cost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_creation,
            # A transcript is history: price each turn at the rate in effect
            # when it ran, not at today's. `ts` is the same instant this span's
            # start_time gets below; None (no timestamp parsed) falls back to
            # the current rate, matching what start_time does.
            at=ts,
        )
        # Provenance for cost_usd (mirrors CostEngine.process_span on the live
        # path — backfilled spans are pre-priced and never reach it, see
        # ingest.py's was_pre_priced handling, so it's stamped here instead).
        pricing_source = classify_pricing_source(provider, model)
        # Persist the cache read/write split, mirroring the live ingest path
        # (#245). cache_tokens = cache-READ only; cache_write_tokens =
        # cache-CREATION (priced higher). Collapsing them into one field made
        # the Cost table show CACHE W = 0 and a CACHE R that was actually
        # read+write. See models.py NormalizedSpan + Critical Rule on cache.

        agent_id = _agent_id_from_cwd(cwd)
        start_time = ts

        # Per-message content (opt-in, gated by [capture]). Default-off leaves
        # llm_attrs carrying provenance only — the ingest source and the call
        # id below, no content. Keys match GenAIAttributes so downstream consumers (and
        # alert content-stripping) treat backfilled content like live content.
        # `tj.call_id` names the API CALL this span observes, so a second
        # observer of the same call can be recognised rather than counted
        # again. The assistant message key is that name on this path: it is
        # stable across resumes and replays (see core.usage), which is exactly
        # what the span_id is already derived from.
        llm_attrs: dict = {
            "source": _CLAUDE_CODE_SOURCE,
            TjAttributes.CALL_ID: message_key,
        }
        if capture.prompts and pending_prompt.strip():
            llm_attrs[GenAIAttributes.PROMPT_CONTENT] = pending_prompt
        if capture.prompts:
            # Only commit the sentinel once a real lookup happened. A record
            # without `cwd` can't resolve anything, so leaving the sentinel at
            # None lets a later record that DOES carry cwd try again.
            if claude_md_text is None and cwd is not None:
                claude_md_text = _read_project_claude_md(cwd)
            if claude_md_text:
                llm_attrs.update(_system_prefix_attrs(claude_md_text))
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
            sub_agent_type=span_sub_agent_type,
            session_id=sid_str,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_tokens=cache_read,
            cache_write_tokens=cache_creation,
            cost_usd=cost,
            pricing_source=pricing_source,
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
                    sub_agent_type=span_sub_agent_type,
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
    if earliest is None:
        # Every record in this file was undated, so there is no session to
        # describe — returning one would invent both its start and its extent.
        return None
    started_at = earliest
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
        transcript_mtime=transcript_mtime,
        records_undated=records_undated,
        first_user_prompt=first_user_prompt,
        dominant_model=(
            max(_model_tokens, key=lambda m: _model_tokens[m]) if _model_tokens else None
        ),
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


def _status_for_backfilled_session(transcript_mtime: datetime | None) -> str:
    """Derive a backfilled Claude Code session's lifecycle status from its
    transcript's on-disk mtime, instead of hardcoding "completed".

    Reuses `SESSION_STALE_THRESHOLD` -- the SAME window
    `SessionRecord.status_with_transcript_mtime` already uses to rescue a live
    session from a misleadingly stale span signal -- so there is exactly one
    definition of "is this transcript still being written to" in the codebase.
    A transcript modified within the threshold is still being appended to by a
    live terminal -> "active" (a non-terminal status; read-time
    `SessionRecord.status_at` still degrades it to idle/stale as the gap since
    `ended_at` grows, and the mtime-rescue path stays available on every later
    read). A transcript that has gone quiet past the threshold (or one we
    couldn't stat at all) has nothing left to rescue it here -> "completed".
    """
    from tokenjam.utils.time_parse import utcnow
    if transcript_mtime is not None and utcnow() - transcript_mtime <= SESSION_STALE_THRESHOLD:
        return "active"
    return "completed"


def session_record_from_parsed(
    parsed: ParsedSession, plan_tier: str = "unknown",
) -> SessionRecord:
    return SessionRecord(
        session_id=parsed.session_id,
        agent_id=parsed.agent_id,
        started_at=parsed.started_at,
        ended_at=parsed.ended_at,
        conversation_id=parsed.session_id,
        status=_status_for_backfilled_session(parsed.transcript_mtime),
        total_cost_usd=parsed.total_cost_usd,
        input_tokens=parsed.total_input_tokens,
        output_tokens=parsed.total_output_tokens,
        cache_tokens=parsed.total_cache_tokens,
        tool_call_count=parsed.tool_call_count,
        error_count=0,
        plan_tier=plan_tier,
        # This function ONLY ever parses a Claude Code transcript — a literal
        # fact, not a heuristic (contrast the live path, which has to
        # classify an ambiguous agent_id; see core.agent_kind). Matches
        # `agent_kind.CODING_AGENT_GROUPS`'s "claude-code" spelling exactly —
        # not "claude_code" — so a value this function stamps and one the
        # live path derives via `classify_agent_kind` are the SAME string.
        source="claude-code",
        task_statement_hash=hash_task_statement(parsed.first_user_prompt),
        dominant_model=parsed.dominant_model,
    )


def session_totals_delta(
    parsed: ParsedSession, plan_tier: str, inserted: list[NormalizedSpan],
) -> SessionRecord:
    """The session row this file's write ADDS, not the session's whole life.

    A Claude Code session is split across files sharing one session_id (the
    main-thread transcript plus each `subagents/agent-*.jsonl`), and a backfill
    ingests them one at a time. `session_record_from_parsed` describes what ONE
    file saw, so writing it with replace semantics leaves the row holding the
    last file's totals while `SUM(spans)` holds all of them — the drift
    `session_cost_drift` reports. Summing over the spans this write actually
    INSERTED and adding that (`upsert_session(..., accumulate_totals=True)`)
    makes the row agree with the spans by construction, and makes a re-run of
    the same file add zero, since nothing is inserted the second time.

    Summed over the inserted spans rather than over the parsed file for the
    same reason: a span suppressed as another observer's duplicate, or already
    present, contributed nothing to `SUM(spans)` and must contribute nothing
    here either.
    """
    return replace(
        session_record_from_parsed(parsed, plan_tier),
        total_cost_usd=sum(s.cost_usd or 0.0 for s in inserted),
        input_tokens=sum(s.input_tokens or 0 for s in inserted),
        output_tokens=sum(s.output_tokens or 0 for s in inserted),
        cache_tokens=sum(s.cache_tokens or 0 for s in inserted),
        # Not carried by `session_record_from_parsed` at all, which is its own
        # under-report: cache WRITES are the priciest bucket.
        cache_write_tokens=sum(s.cache_write_tokens or 0 for s in inserted),
        tool_call_count=sum(1 for s in inserted if s.tool_name),
        error_count=sum(1 for s in inserted if s.status_code == SpanStatus.ERROR),
    )


# --- Ingest -----------------------------------------------------------------

# New spans accumulated before a columnar bulk-append flush. Batching the INSERT
# across sessions amortizes `read_json`'s per-call fixed cost — measured ~1.5×
# faster than a flush-per-session on a history of thousands of tiny sessions. A
# batch is a few MB of NDJSON, streamed to a temp file, so memory stays flat
# regardless of total history size. (Existence-check + counting stay PER SESSION
# so `result` — and any live progress display reading it — advances monotonically
# instead of sitting at zero until the final flush.)
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
    db, parsed: ParsedSession, plan_tier: str, reingest: bool, result: BackfillResult,
    scan: DuplicateScan | None = None,
) -> None:
    """Per-session insert path (reingest, no-conn fallback, and the bulk-flush
    error fallback). Mirrors the historical per-file behavior including the
    `files_failed` / `sample_errors` accounting on a DB error."""
    try:
        inserted, retagged = _insert_session_idempotent(
            db, parsed, plan_tier=plan_tier, reingest=reingest, scan=scan,
        )
    except Exception as exc:
        result.files_failed += 1
        if len(result.sample_errors) < 5:
            result.sample_errors.append(f"{parsed.session_id}: {exc}")
        return
    _record_insert_outcome(result, parsed, inserted, retagged)


#: Attribute keys `parse_claude_code_session` always sets, regardless of
#: `[capture]` — the provenance tag and the call-identity key (see the
#: `llm_attrs` construction in that function). A span whose `attributes`
#: contains nothing BEYOND these carries no content worth overlaying, so
#: checking the size of this set is a cheap pre-filter before ever building
#: the (comparatively expensive) `json_merge_patch` candidate payload.
_BASELINE_ATTRIBUTE_KEYS = frozenset({"source", TjAttributes.CALL_ID})


def _dedup_new_spans(
    conn, parsed: ParsedSession, scan: DuplicateScan | None = None,
    capture: CaptureConfig | None = None,
) -> tuple[list[NormalizedSpan], list[NormalizedSpan]]:
    """Partition `parsed`'s spans into (new, overlay_candidates) — a cheap,
    indexed, chunked existence check. Kept PER SESSION (not batched) so
    `spans_ingested` can be counted the moment a session is processed, giving
    the progress callback monotonically-increasing counts while the actual
    columnar INSERT is deferred to a batched flush.

    `new` are spans not already present in the DB. `overlay_candidates` are
    spans that ARE already present but whose re-parse resolved something new
    the stored row is missing — either half of the set-based overlay
    `bulk_overlay_span_attrs` performs:
      - `sub_agent_id`/`sub_agent_type` this file's transcript can supply —
        e.g. a row inserted before migration 19 (`sub_agent_type`) existed,
        or before this file's sidecar (`agent-<id>.meta.json`) was written.
      - Captured content (`gen_ai.prompt.content` etc.) when `[capture]` is
        ON now but was off when the row was first ingested — gated on
        `capture` actually having a toggle enabled AND the re-parsed span
        carrying more than the baseline provenance keys, so a plain re-run
        with capture off (the common case) never re-queues every existing
        span just to ship a no-op payload.
    The caller batches these into a set-based overlay UPDATE rather than
    touching them here, for the same reason new spans are batched: per-span
    UPDATEs are the ~350×-slower path this bulk branch exists to avoid.

    Also drops calls the LIVE path already recorded (`scan` carries the
    running per-call tally across this run's files) — see
    `_drop_calls_another_source_recorded`."""
    existing = _existing_span_ids(conn, [s.span_id for s in parsed.spans])
    new_spans = [s for s in parsed.spans if s.span_id not in existing]
    new_spans = _drop_calls_another_source_recorded(
        conn, parsed.session_id, new_spans, scan,
    )
    capture_on = capture is not None and (
        capture.prompts or capture.completions or capture.tool_inputs
    )
    overlay_candidates = [
        s for s in parsed.spans
        if s.span_id in existing and (
            s.sub_agent_id or s.sub_agent_type
            or (capture_on and set(s.attributes) - _BASELINE_ATTRIBUTE_KEYS)
        )
    ]
    return new_spans, overlay_candidates


@dataclass
class DuplicateScan:
    """Per-run state for cross-source duplicate suppression.

    Threaded through one backfill run so two questions are each asked once
    rather than once per file:

    * `other_source_present` — is there any non-backfill observation in this
      store at all? A duplicate needs two observers, so on a machine that has
      only ever backfilled every per-session lookup would be wasted work.
    * `suppressed` — how many observations of each call this run has already
      dropped, keyed (session_id, call fingerprint). A suppressed span is never
      stored, so without this tally the same duplicate budget would be spent
      again on the next file of the same session and a genuinely repeated call
      would be dropped.
    """
    other_source_present: bool | None = None
    suppressed: dict[tuple[str, str], int] = field(default_factory=dict)


def _drop_calls_another_source_recorded(
    conn,
    session_id: str,
    new_spans: list[NormalizedSpan],
    scan: DuplicateScan | None = None,
) -> list[NormalizedSpan]:
    """Drop LLM spans describing a call the live path already recorded.

    A session that ran while `tj serve` was up is observed twice: once live, as
    it happened, and again here when its transcript is parsed. The two
    observations mint different span_ids, so the existence check above cannot
    see the overlap and every cost figure prices the call twice. They agree on
    the call's billed shape, which is what `accounting.call_fingerprint` names.

    Only ever collapses ACROSS ingest sources, and only up to the number of
    observations the other source actually recorded (`duplicate_budget`) — two
    identically-shaped calls in one session are two real calls, and dropping
    one would under-report spend. `scan` carries that budget across the files
    of one session, which arrive as separate `ParsedSession`s.

    Best-effort: any lookup failure keeps every span. Tool and marker spans are
    never candidates — they carry no money and no billed shape.
    """
    if conn is None or not session_id or not new_spans:
        return new_spans

    from tokenjam.core.db import (
        has_spans_from_another_source, stored_observations_by_call,
    )
    from tokenjam.core.optimize import accounting

    candidates = [
        s for s in new_spans
        if s.name == GenAIAttributes.SPAN_LLM_CALL and s.model and not s.tool_name
    ]
    if not candidates:
        return new_spans
    if scan is not None:
        if scan.other_source_present is None:
            try:
                scan.other_source_present = has_spans_from_another_source(
                    conn, _CLAUDE_CODE_SOURCE,
                )
            except Exception as exc:
                logger.warning("duplicate-observation probe skipped: %s", exc)
                scan.other_source_present = False
        if not scan.other_source_present:
            return new_spans
    try:
        stored = stored_observations_by_call(conn, session_id)
    except Exception as exc:  # never let the guard break the ingest
        logger.warning("duplicate-observation check skipped: %s", exc)
        return new_spans
    if not stored:
        return new_spans

    tally = scan.suppressed if scan is not None else {}
    dropped: set[str] = set()
    for span in candidates:
        fingerprint = accounting.call_fingerprint(
            session_id, span.model,
            span.input_tokens or 0, span.output_tokens or 0,
            span.cache_tokens or 0, span.cache_write_tokens or 0,
        )
        by_source = stored.get(fingerprint)
        if not by_source:
            continue
        key = (session_id, fingerprint)
        already = tally.get(key, 0)
        if accounting.duplicate_budget(by_source, _CLAUDE_CODE_SOURCE, already) <= 0:
            continue
        tally[key] = already + 1
        dropped.add(span.span_id)

    if not dropped:
        return new_spans
    logger.debug(
        "backfill skipped %d span(s) the live path already recorded (session=%s)",
        len(dropped), session_id,
    )
    return [s for s in new_spans if s.span_id not in dropped]


def _flush_pending_spans(db, pending: list[NormalizedSpan]) -> None:
    """Columnar bulk-append of the accumulated new spans (deferred from the
    per-session loop so many small sessions share one `read_json` scan). On the
    rare catastrophic append error, degrade to per-row inserts so the ingest
    keeps making progress rather than aborting — every pending span was already
    dedup'd as new, so nothing double-inserts."""
    if not pending:
        return
    try:
        db.bulk_insert_spans(pending)
    except Exception:
        logger.warning(
            "bulk span append failed; falling back to per-row insert", exc_info=True
        )
        for span in pending:
            try:
                db.insert_span(span)
            except Exception:
                pass


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
    # Cross-source duplicate-suppression state, shared by every file this run
    # touches (see DuplicateScan).
    duplicate_scan = DuplicateScan()

    # A fresh full backfill (the ~8min/5.6GB hot path) dedups + counts + upserts
    # each session eagerly (so `result` — and any live progress display reading it
    # — advances monotonically), but DEFERS the actual span INSERT into a
    # cross-session buffer flushed through the columnar bulk-append. `reingest`
    # keeps the fully per-session path (its per-span attribute overlay is not a
    # bulk op); a backend without a `conn` (defensive) also falls back per session.
    conn = getattr(db, "conn", None)
    use_bulk = conn is not None and not reingest
    pending: list[NormalizedSpan] = []
    pending_ids: set[str] = set()
    # Existing spans this run resolved something new for — sub_agent_id/
    # sub_agent_type, or captured content (see `_dedup_new_spans`'s
    # `overlay_candidates`) — queued for a batched additive UPDATE alongside
    # the new-span flush below. `pending_overlay_ids` is belt-and-braces
    # against queuing the same span_id twice, matching `pending_ids` above.
    pending_overlay: list[tuple[str, str | None, str | None, dict | None]] = []
    pending_overlay_ids: set[str] = set()
    # Session-total deltas wait for the spans they describe. Nothing here has
    # transaction control — every statement auto-commits — so the only lever on
    # a mid-run interruption is ORDER, and the two orders fail very differently:
    #
    #   delta first  → the row counts spans that were still only in `pending`
    #                  and died with the process. The next run re-parses those
    #                  files, finds the span_ids genuinely absent, calls them
    #                  new, and adds the SAME delta a second time. The drift
    #                  compounds on every interrupted run and nothing heals it:
    #                  the end-of-run recompute never ran.
    #   spans first  → the row is briefly LOW. The spans are durable, so the
    #                  next run dedups them away (delta zero) and the
    #                  unconditional `recompute_session_totals_from_spans`
    #                  below rewrites the row from `SUM(spans)`.
    #
    # An undercount that self-heals beats an overcount that accumulates, so the
    # deltas ride with their spans and are applied only after the append lands.
    pending_deltas: list[SessionRecord] = []

    def _flush_pending() -> None:
        if pending:
            _flush_pending_spans(db, pending)
            pending.clear()
            pending_ids.clear()
        if pending_overlay:
            overlay = getattr(db, "bulk_overlay_span_attrs", None)
            if overlay is not None:
                try:
                    changed = overlay(pending_overlay)
                except Exception:
                    # Best-effort: the overlay is a repair of an already-stored
                    # row, never a reason to fail a backfill that otherwise
                    # succeeded. The next run offers the same candidates again.
                    logger.warning("subagent overlay flush failed", exc_info=True)
                else:
                    result.spans_retagged += changed
                    # These spans were provisionally counted as "skipped
                    # existing" per-session (see `_record_insert_outcome`)
                    # before we knew how many the overlay would actually
                    # change; move the ones that did change into `retagged` so
                    # the two counts stay mutually exclusive and sum to the
                    # session's total spans, same invariant `--reingest` keeps.
                    result.spans_skipped_existing -= changed
            pending_overlay.clear()
            pending_overlay_ids.clear()
        for delta in pending_deltas:
            try:
                db.upsert_session(delta, accumulate_totals=True)
            except Exception:
                # The end-of-run recompute reconciles every seen session from
                # SUM(spans), so a dropped delta costs accuracy only if this
                # run also dies before it — the same self-healing undercount.
                logger.warning(
                    "session total delta failed for %s; will reconcile from spans",
                    delta.session_id, exc_info=True,
                )
        pending_deltas.clear()

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
        result.records_undated += parsed.records_undated
        if result.earliest is None or parsed.started_at < result.earliest:
            result.earliest = parsed.started_at
        if result.latest is None or parsed.ended_at > result.latest:
            result.latest = parsed.ended_at

        if use_bulk:
            try:
                new_spans, overlay_candidates = _dedup_new_spans(
                    conn, parsed, duplicate_scan, capture=capture,
                )
            except Exception as exc:
                result.files_failed += 1
                if len(result.sample_errors) < 5:
                    result.sample_errors.append(f"{parsed.session_id}: {exc}")
                continue
            for span in overlay_candidates:
                if span.span_id in pending_overlay_ids:
                    continue
                pending_overlay_ids.add(span.span_id)
                pending_overlay.append(
                    (span.span_id, span.sub_agent_id, span.sub_agent_type, span.attributes)
                )
            # Queue the new spans for the batched INSERT, but count them NOW so the
            # progress callback sees an increasing `spans_ingested` instead of a
            # flat zero that only jumps at the final flush. The `pending_ids` guard
            # keeps the same span_id out of one `read_json` batch twice (span_ids
            # are session-scoped and unique in practice — this is belt-and-braces).
            queued_spans: list[NormalizedSpan] = []
            for span in new_spans:
                if span.span_id in pending_ids:
                    continue
                pending_ids.add(span.span_id)
                pending.append(span)
                queued_spans.append(span)
            # The session row gains exactly what this file's spans add — the
            # spans QUEUED, not the ones merely found new, so a span_id the
            # `pending_ids` guard dropped is not counted for a row it will
            # never appear in. Queued alongside those spans and applied by
            # `_flush_pending` once the append lands, so the row and
            # `SUM(spans)` agree, and a re-run of the same file adds nothing.
            pending_deltas.append(
                session_totals_delta(parsed, plan_tier, queued_spans)
            )
            _record_insert_outcome(result, parsed, len(queued_spans), 0)
            if len(pending) >= _BULK_FLUSH_SPAN_TARGET or len(pending_overlay) >= _BULK_FLUSH_SPAN_TARGET:
                _flush_pending()
        else:
            _apply_session(db, parsed, plan_tier, reingest, result, duplicate_scan)

        if progress is not None:
            try:
                progress(parsed, result)
            except Exception:
                pass

    _flush_pending()

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
    #
    # Only ever runs on an UNBOUNDED pass, because the DELETE is only safe when
    # `keep_by_session[sid]` really is the session's COMPLETE span set:
    #   * `max_sessions` (the quickstart cap) stops mid-walk, so a session's
    #     later files may never be parsed.
    #   * `since` filters files by mtime AND by parsed `ended_at`, so a session
    #     straddling the window boundary — a long-running conversation whose main
    #     transcript is still being appended while its `subagents/` files finished
    #     days ago — yields only its in-window files. The out-of-window siblings'
    #     already-ingested spans would then look like stale-scheme orphans and be
    #     DELETED. Verified: a two-file session re-ingested with `--since 2d`
    #     after the subagent file aged out lost that file's span.
    # The full `tj backfill claude-code` path (also used by onboard) is
    # unwindowed and still does the self-healing.
    reconcile = getattr(db, "reconcile_backfill_spans", None)
    windowed = since is not None or max_sessions is not None
    if reconcile is not None and not windowed and keep_by_session:
        try:
            purged = reconcile(keep_by_session, _CLAUDE_CODE_SOURCE)
            result.spans_stale_purged = purged
        except Exception as exc:  # never let reconciliation break the ingest
            logger.warning("stale-span reconciliation skipped: %s", exc)

    # Reconcile each touched session row to the SUM of its spans. The per-file
    # upsert above now ADDS its own spans' totals rather than replacing the row
    # (see session_totals_delta), so a run against a current DB finds nothing to
    # change -- this is no longer the mechanism that makes multi-file sessions
    # add up, it is the repair for the two things the write cannot cover:
    #   * the stale-scheme purge just above DELETED rows, moving SUM(spans)
    #     under a row nobody rewrote;
    #   * a row written by an older build, whose replacing per-file upserts left
    #     it holding only the last file's totals, is a wrong base for any later
    #     accumulation.
    # Idempotent, so it stays safe to run unconditionally.
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

    # Refresh the statusline's cheap top-driver cache — best-effort, never
    # blocks or fails the ingest. The statusline is DB-free and can't compute
    # recurring-inclusion attribution itself; this backfill path already holds
    # the connection + capture flags, so it hands the result off as a small
    # on-disk artifact the statusline can stat+read.
    if conn is not None:
        from tokenjam.core.attribution_cache import refresh_attribution_cache
        refresh_attribution_cache(conn, capture)

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
    db, parsed: ParsedSession, plan_tier: str = "unknown", reingest: bool = False,
    scan: DuplicateScan | None = None,
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
      - `sub_agent_id`   — re-tags history ingested before that column existed.
      - `sub_agent_type` — same, for the stable subagent identity (migration 19);
        re-running over unchanged transcripts populates it on already-ingested
        spans and is otherwise a no-op (the value is derived from the on-disk
        sidecar, so an unchanged transcript always re-derives the same value).
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
        written: list[NormalizedSpan] = []
        for span in parsed.spans:
            try:
                db.insert_span(span)
                written.append(span)
                inserted += 1
            except Exception:
                continue
        db.upsert_session(
            session_totals_delta(parsed, plan_tier, written), accumulate_totals=True,
        )
        return inserted, retagged

    span_ids = [s.span_id for s in parsed.spans]
    existing = _existing_span_ids(conn, span_ids)
    new_spans = _drop_calls_another_source_recorded(
        conn, parsed.session_id,
        [s for s in parsed.spans if s.span_id not in existing],
        scan,
    )
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
            # COALESCE, not overwrite: a re-derived value is always identical
            # for an unchanged transcript, but if the sidecar has since gone
            # missing (pruned) a re-parse yields None — COALESCE keeps
            # whatever the row already resolved rather than clobbering it back
            # to NULL. Matches the bulk overlay's additive semantics (see
            # `_SUBAGENT_OVERLAY_MATCH_PREDICATE` in `core/db.py`).
            conn.execute(
                "UPDATE spans SET "
                "sub_agent_id = COALESCE(sub_agent_id, $1), "
                "sub_agent_type = COALESCE(sub_agent_type, $2), "
                "attributes = $3 WHERE span_id = $4",
                [span.sub_agent_id, span.sub_agent_type,
                 json.dumps(merged_attrs), span.span_id],
            )
            retagged += 1

    db.upsert_session(
        session_totals_delta(parsed, plan_tier, new_spans), accumulate_totals=True,
    )
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
