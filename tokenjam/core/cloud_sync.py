"""
The bridge, sender side: forward spans, sessions and `session_commits` from
the local DuckDB store to TokenJam Cloud (ledger W5; contracts §3, §6, §9).

`tj init --cloud <key> --org <org>` writes `[cloud]` once; from then on the
daemon calls :func:`start_sync` on a schedule and `tj init --cloud` itself runs
one :func:`run_sync` for the initial push. Both read the same three streams
and the same resume state:

* **spans** go to the existing `POST /api/v1/spans` as OTLP JSON. Every
  stored span is re-encoded through the same attribute names
  `tokenjam.otel.otlp_parsing.parse_otlp_span` reads, so the Cloud parser
  (which IS that function) rebuilds the span the local store holds. The
  contracts §3 attributes ride as resource attributes (install id, host) and
  span attributes (the session's repo / branch / developer columns), nothing
  new on the wire.
* **sessions** and **session_commits** go to `POST /api/v1/ledger/sessions`
  as `{"sessions": [...], "session_commits": [...]}`, each list at most
  `BATCH_SIZE` rows, idempotent by primary key on the receiver.

**Resumable.** `~/.tj/cloud_sync.json` holds one high-water mark per stream
in ARRIVAL order, never event order: `spans.ingested_at` (stamped on insert),
`sessions.updated_at` (stamped on insert and by every session writer, so a
session whose totals grew, that was closed, or whose plan tier was stamped
later is re-sent) and `session_commits.matched_at` (set at write time and
re-stamped by a confidence upgrade). Event time would lose rows: a backfill
or the daemon's transcript catch-up inserts spans and sessions OLDER than
everything already forwarded. The mark carries the ids already sent AT its
timestamp so a resume can re-read from it inclusive without re-sending
them. State is written after every acknowledged batch, never before, so a
crash mid-pass re-sends at most one batch, which the receiver dedupes. The
marks are scoped to the org AND the database file they describe; either
changing starts them over.

**Content never crosses by default.** Prompt, completion and tool content is
stripped unless the local `[capture]` toggles keep it AND
`[cloud] forward_content` is true (contracts §6). The strip is the same
`strip_captured_content` the local ingest gates on, followed by a leaf-name
sweep for content keys that list does not name, so the promise does not
depend on the receiver.

**Never blocks local ingest, never takes the database down.** The forwarder
runs on its own thread with its own backend, swallows every error into the
state file's `last_error`, and classifies a DuckDB fatal the way the other
daemon jobs do (Critical Rule 45): recorded where it is recognised, recovered
in a `finally` off the process-wide record.

**A 401 disables the bridge** with a reason `tj status` and `tj doctor`
print, rather than retrying a dead key every five minutes forever;
`tj init --cloud <key>` clears it. 5xx, 429 and network failures back off and
resume on the next pass. Any other 4xx is a batch the receiver refused: the
batch is split until the refused record is isolated, then skipped and counted
in `rejected` so one bad row can never wedge the stream.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.ingest import strip_captured_content
from tokenjam.core.models import (
    SESSION_CONTEXT_FIELDS,
    NormalizedSpan,
    SessionCommit,
    SessionRecord,
)
from tokenjam.core.repo_context import ensure_install_id, repo_name_from_url, tj_home
from tokenjam.otel.semconv import GenAIAttributes, ResourceAttributes, TjAttributes
from tokenjam.utils.time_parse import utcnow

logger = logging.getLogger(__name__)

#: Rows per request, both for the OTLP body and for each ledger list
#: (contracts §6: "batched ≤500").
BATCH_SIZE = 500

#: Route paths on the Cloud API (contracts §6).
SPANS_PATH = "/api/v1/spans"
LEDGER_SESSIONS_PATH = "/api/v1/ledger/sessions"

#: Header names the Cloud ingest auth reads.
ORG_HEADER = "X-TokenJam-Org"

#: How often the daemon runs a pass.
SYNC_INTERVAL_MINUTES = 5

#: Retry schedule for 5xx / 429 / network failures inside one pass, in
#: seconds between attempts. Exhausting it ends the pass; the high-water mark
#: is untouched, so the next pass resumes exactly there.
RETRY_BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 4.0)

#: Per-request timeout. The receiver batches its writes, so a 500-row body is
#: one round trip; this only has to outlast a slow cold start.
REQUEST_TIMEOUT_S = 30.0

#: Contracts §9, verbatim. `tj init --cloud` prints these before the first
#: byte leaves; `tests/unit/test_cloud_sync.py` pins that it does.
EMISSION_LEAVES: tuple[str, ...] = (
    "token counts",
    "model names",
    "cost",
    "timestamps",
    "tool names",
    "file paths touched",
    "session / repo / branch / commit identifiers",
    "hashed developer id",
    "git author email",
)
EMISSION_NEVER_BY_DEFAULT: tuple[str, ...] = (
    "prompt text",
    "completions",
    "tool outputs",
    "file contents",
    "diffs",
    "secrets",
)

#: Ingest keys Cloud mints (`api/app/security.py::generate_ingest_key`).
INGEST_KEY_PREFIX = "tj_live_"

#: Leaf names that mean "this holds what was said". Swept off every span
#: attribute when content forwarding is off, on top of the explicit list
#: `strip_captured_content` pops, so a vendor key the list never named
#: (`llm.prompts`, `input.value`) is caught by shape rather than by name.
_CONTENT_LEAVES = frozenset({
    "content", "prompt", "prompts", "completion", "completions",
    "text", "message", "messages", "body", "input", "output",
})
_CONTENT_KEYS = frozenset({"input.value", "output.value"})

#: Everything stripped: the gate the forwarder applies unless the user turned
#: `forward_content` on.
_CAPTURE_NONE = CaptureConfig(
    prompts=False, completions=False, tool_inputs=False, tool_outputs=False,
)

#: One pass at a time per process. The daemon's interval job and its
#: startup kick both dispatch a thread and return; a receiver that is slow
#: or backing off can keep one pass alive past the next tick, and two passes
#: over the same marks would re-send batches and race on the state file.
_PASS_LOCK = threading.Lock()

_KIND_MAP = {"internal": 1, "server": 2, "client": 3, "producer": 4, "consumer": 5}
_STATUS_MAP = {"unset": 0, "ok": 1, "error": 2}


# --- Resume state -------------------------------------------------------------

def state_path() -> Path:
    return tj_home() / "cloud_sync.json"


@dataclass
class StreamMark:
    """One stream's high-water mark: the last timestamp acknowledged and the
    ids sent at exactly that timestamp (a resume re-reads from `hwm`
    inclusive and skips these, so a timestamp shared by several rows can be
    crossed without re-sending or skipping any of them)."""
    hwm: str | None = None
    in_flight_ids: list[str] = field(default_factory=list)

    def hwm_dt(self) -> datetime | None:
        if not self.hwm:
            return None
        try:
            return datetime.fromisoformat(self.hwm)
        except ValueError:
            return None


@dataclass
class SyncState:
    spans: StreamMark = field(default_factory=StreamMark)
    sessions: StreamMark = field(default_factory=StreamMark)
    session_commits: StreamMark = field(default_factory=StreamMark)
    spans_sent: int = 0
    sessions_sent: int = 0
    commits_sent: int = 0
    rejected: int = 0
    last_success_at: str | None = None
    last_attempt_at: str | None = None
    last_error: str | None = None
    #: Set on a 401; the forwarder does nothing while it is set.
    disabled_reason: str | None = None
    disabled_at: str | None = None
    #: The org the marks belong to, and the database file they were read
    #: from. Either changing resets every mark: another org holds none of
    #: the rows, and another store has its own arrival order.
    org_id: str | None = None
    db_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SyncState":
        def _mark(v: Any) -> StreamMark:
            if not isinstance(v, Mapping):
                return StreamMark()
            ids = v.get("in_flight_ids")
            return StreamMark(
                hwm=v.get("hwm") if isinstance(v.get("hwm"), str) else None,
                in_flight_ids=[str(i) for i in ids] if isinstance(ids, list) else [],
            )

        def _int(v: Any) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        def _str(v: Any) -> str | None:
            return v if isinstance(v, str) and v else None

        return cls(
            spans=_mark(raw.get("spans")),
            sessions=_mark(raw.get("sessions")),
            session_commits=_mark(raw.get("session_commits")),
            spans_sent=_int(raw.get("spans_sent")),
            sessions_sent=_int(raw.get("sessions_sent")),
            commits_sent=_int(raw.get("commits_sent")),
            rejected=_int(raw.get("rejected")),
            last_success_at=_str(raw.get("last_success_at")),
            last_attempt_at=_str(raw.get("last_attempt_at")),
            last_error=_str(raw.get("last_error")),
            disabled_reason=_str(raw.get("disabled_reason")),
            disabled_at=_str(raw.get("disabled_at")),
            org_id=_str(raw.get("org_id")),
            db_path=_str(raw.get("db_path")),
        )


def load_state(path: Path | None = None) -> SyncState:
    """The persisted resume state, or a fresh one when the file is absent or
    unreadable (a corrupt state file costs one re-send of history, which the
    receiver dedupes; it never costs the pass)."""
    target = path or state_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return SyncState()
    return SyncState.from_dict(raw) if isinstance(raw, Mapping) else SyncState()


def save_state(state: SyncState, path: Path | None = None) -> None:
    """Atomic write; best effort (a state that cannot be written means the
    next pass re-sends one batch more, nothing worse)."""
    target = path or state_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        logger.debug("could not persist cloud sync state at %s", target, exc_info=True)


def storage_identity(config: TjConfig) -> str:
    """The database file the marks describe, resolved the way `DuckDBBackend`
    opens it, so two configs naming the same file share one set of marks
    and two files never do."""
    return str(Path(config.storage.path).expanduser().resolve())


def reset_state(*, org_id: str, db_path: str | None = None, path: Path | None = None) -> SyncState:
    """What `tj init --cloud` does to the resume state. For the org and store
    the marks already describe, only the receiver's disable and the last
    error are cleared (a re-run with a fresh key resumes, it does not re-send
    history); for any other org or store the marks start over. Written
    immediately."""
    existing = load_state(path)
    if existing.org_id == org_id and (db_path is None or existing.db_path == db_path):
        existing.disabled_reason = None
        existing.disabled_at = None
        existing.last_error = None
        state = existing
    else:
        state = SyncState(org_id=org_id, db_path=db_path)
    save_state(state, path)
    return state


# --- Key parsing ----------------------------------------------------------------

def parse_ingest_key(raw: str) -> str:
    """Validate an ingest key as Cloud mints it. Returns the trimmed key or
    raises ValueError naming what is wrong. The key carries no org id (it is
    `tj_live_` plus random bytes), which is why `tj init --cloud` needs
    `--org` beside it."""
    key = (raw or "").strip()
    if not key.startswith(INGEST_KEY_PREFIX):
        raise ValueError(
            f"an ingest key starts with {INGEST_KEY_PREFIX!r}; copy it from "
            "Cloud's Connect screen"
        )
    if len(key) < len(INGEST_KEY_PREFIX) + 16 or any(c.isspace() for c in key):
        raise ValueError("the ingest key looks truncated; copy the whole value")
    return key


# --- Wire encoding ---------------------------------------------------------------

def _otlp_value(value: Any) -> dict[str, Any] | None:
    """One OTLP AttributeValue. `None` for a value OTLP cannot carry."""
    if value is None:
        return None
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        vals = [v for v in (_otlp_value(x) for x in value) if v is not None]
        return {"arrayValue": {"values": vals}}
    if isinstance(value, Mapping):
        kvs = []
        for k, v in value.items():
            enc = _otlp_value(v)
            if enc is not None:
                kvs.append({"key": str(k), "value": enc})
        return {"kvlistValue": {"values": kvs}}
    return {"stringValue": str(value)}


def _otlp_attrs(attrs: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for k, v in attrs.items():
        enc = _otlp_value(v)
        if enc is not None:
            out.append({"key": str(k), "value": enc})
    return out


def _ns(dt: datetime | None) -> str:
    return str(int(dt.timestamp() * 1e9)) if dt else "0"


def _iso(dt: datetime | None) -> str | None:
    """ISO 8601 in UTC. DuckDB hands TIMESTAMPTZ back in the local zone; the
    wire and the state file carry one zone so a mark written on one machine
    setting reads the same after a timezone change."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return dt.astimezone(timezone.utc).isoformat()


def _sweep_content_keys(attrs: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        lowered = str(key).lower()
        leaf = lowered.rsplit(".", 1)[-1]
        if leaf in _CONTENT_LEAVES or lowered in _CONTENT_KEYS:
            continue
        out[key] = value
    return out


def content_gate(config: TjConfig) -> CaptureConfig:
    """The capture posture the forwarder applies: everything stripped unless
    `[cloud] forward_content` is on, in which case the local `[capture]`
    toggles decide (contracts §6: local capture AND forward_content)."""
    if not config.cloud.forward_content:
        return _CAPTURE_NONE
    return config.capture


def _all_off(capture: CaptureConfig) -> bool:
    return not (capture.prompts or capture.completions
                or capture.tool_inputs or capture.tool_outputs)


def _all_on(capture: CaptureConfig) -> bool:
    return (capture.prompts and capture.completions
            and capture.tool_inputs and capture.tool_outputs)


#: The content keys `strip_captured_content` governs toggle by toggle. After
#: it has run, these carry exactly what the local toggles allow; every OTHER
#: content-shaped key is one no toggle names, so it crosses only when the
#: user has turned every toggle on.
_GOVERNED_CONTENT_KEYS = frozenset({
    GenAIAttributes.PROMPT_CONTENT, GenAIAttributes.COMPLETION_CONTENT,
    GenAIAttributes.TOOL_INPUT, GenAIAttributes.TOOL_OUTPUT,
    TjAttributes.REQUEST_TOOLS, TjAttributes.SYSTEM_PREFIX_CONTENT,
    TjAttributes.SYSTEM_PREFIX_SAMPLE, TjAttributes.SYSTEM_PREFIX_HASH,
    TjAttributes.SYSTEM_PREFIX_LENGTH,
})


def _scrub(attrs: Mapping[str, Any], capture: CaptureConfig) -> dict[str, Any]:
    """The toggles decide the governed keys; the sweep decides the rest.
    A vendor key no toggle names (`output.value`, `llm.prompts`) is content
    of unknown kind, so a partial configuration (prompts on, completions
    off) cannot vouch for it: it crosses only when all four are on."""
    stripped = strip_captured_content(dict(attrs), capture)
    if _all_on(capture):
        return stripped
    swept = _sweep_content_keys(stripped)
    for k in _GOVERNED_CONTENT_KEYS:
        if k in stripped:
            swept[k] = stripped[k]
    return swept


def _event_ns(e: Mapping[str, Any]) -> str:
    """An event's time as OTLP nanoseconds. Two producers, two spellings:
    the OTLP paths store `time` (already nanoseconds), the in-process SDK
    stores `timestamp` (ISO 8601)."""
    raw = e.get("time")
    if raw not in (None, "", 0, "0"):
        try:
            return str(int(raw))
        except (TypeError, ValueError):
            pass
    stamp = e.get("timestamp")
    if isinstance(stamp, str) and stamp:
        try:
            dt = datetime.fromisoformat(stamp)
        except ValueError:
            return "0"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return _ns(dt)
    return "0"


def _scrub_events(events: Sequence[Mapping[str, Any]] | None,
                  capture: CaptureConfig) -> list[dict[str, Any]]:
    out = []
    for e in events or []:
        entry: dict[str, Any] = {"name": str(e.get("name", "")), "timeUnixNano": _event_ns(e)}
        if not _all_off(capture):
            attrs = e.get("attributes")
            if isinstance(attrs, Mapping):
                entry["attributes"] = _otlp_attrs(_scrub(attrs, capture))
        out.append(entry)
    return out


def session_context_attrs(session: SessionRecord | None) -> dict[str, Any]:
    """The contracts §3 span attributes a session's ledger columns map back
    onto, so the Cloud spans path can persist them onto its own session row.
    Absent columns produce no key, never a placeholder."""
    if session is None:
        return {}
    pairs = (
        (ResourceAttributes.VCS_REPOSITORY_URL_FULL, session.repo_remote),
        (ResourceAttributes.VCS_REPOSITORY_NAME, repo_name_from_url(session.repo_remote)),
        (TjAttributes.REPO_ROOT, session.repo_root),
        (ResourceAttributes.VCS_REF_HEAD_NAME, session.branch_start),
        (TjAttributes.SESSION_BRANCH_END, session.branch_end),
        (ResourceAttributes.VCS_REF_HEAD_REVISION, session.head_sha_start),
        (TjAttributes.SESSION_HEAD_END, session.head_sha_end),
        (TjAttributes.DEVELOPER_ID, session.developer_id),
        (ResourceAttributes.USER_EMAIL, session.user_email),
    )
    return {k: v for k, v in pairs if v}


def span_attributes_for_wire(
    span: NormalizedSpan, session: SessionRecord | None, capture: CaptureConfig,
) -> dict[str, Any]:
    """Every attribute `parse_otlp_span` needs to rebuild this span, content
    gated, plus the §3 context from its session."""
    attrs: dict[str, Any] = dict(span.attributes or {})
    # Structured columns the local store lifted out of the attribute blob at
    # ingest go back under the names the parser reads them from. Sampling
    # params ride the prompts toggle and the tools payload the tool_inputs
    # toggle, exactly as `strip_captured_content` gates them.
    if span.request_params and capture.prompts:
        for k, v in span.request_params.items():
            attrs[f"gen_ai.request.{k}"] = v
    if span.request_tools is not None and capture.tool_inputs:
        attrs[TjAttributes.REQUEST_TOOLS] = span.request_tools
    attrs = _scrub(attrs, capture)

    fixed: dict[str, Any] = {
        GenAIAttributes.AGENT_ID: span.agent_id,
        GenAIAttributes.PROVIDER_NAME: span.provider,
        GenAIAttributes.REQUEST_MODEL: span.model,
        GenAIAttributes.REQUEST_TYPE: span.request_type,
        GenAIAttributes.TOOL_NAME: span.tool_name,
        GenAIAttributes.INPUT_TOKENS: span.input_tokens,
        GenAIAttributes.OUTPUT_TOKENS: span.output_tokens,
        GenAIAttributes.CACHE_READ_TOKENS: span.cache_tokens,
        GenAIAttributes.CACHE_CREATE_TOKENS: span.cache_write_tokens,
        GenAIAttributes.CONVERSATION_ID: span.conversation_id,
        TjAttributes.SESSION_ID: span.session_id,
        TjAttributes.COST_USD: span.cost_usd,
        TjAttributes.BILLING_ACCOUNT: span.billing_account,
        TjAttributes.RUN_ID: span.run_id,
        TjAttributes.PARENT_SESSION_ID: span.parent_session_id,
        TjAttributes.TENANT_ID: span.tenant_id,
        TjAttributes.FEATURE: span.feature,
        TjAttributes.PROMPT_TEMPLATE_ID: span.prompt_template_id,
        TjAttributes.PROMPT_TEMPLATE_VERSION: span.prompt_template_version,
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT_NAME: span.environment,
        ResourceAttributes.SERVICE_VERSION: span.service_version,
        ResourceAttributes.SERVICE_NAMESPACE: span.service_namespace,
        ResourceAttributes.SERVICE_INSTANCE_ID: span.service_instance_id,
        "tokenjam.sub_agent_id": span.sub_agent_id,
        "tokenjam.sub_agent_type": span.sub_agent_type,
    }
    if session is not None and session.plan_tier and session.plan_tier != "unknown":
        fixed[TjAttributes.PLAN_TIER] = session.plan_tier
    if session is not None:
        fixed[ResourceAttributes.SERVICE_NAMESPACE] = (
            fixed[ResourceAttributes.SERVICE_NAMESPACE] or session.service_namespace
        )
        fixed[ResourceAttributes.SERVICE_INSTANCE_ID] = (
            fixed[ResourceAttributes.SERVICE_INSTANCE_ID] or session.service_instance_id
        )
    for k, v in fixed.items():
        if v is not None:
            attrs[k] = v
    for k, v in session_context_attrs(session).items():
        attrs.setdefault(k, v)
    return attrs


def encode_spans_otlp(
    spans: Iterable[NormalizedSpan],
    sessions: Mapping[str, SessionRecord | None],
    *,
    capture: CaptureConfig,
    install_id: str | None,
    host_name: str | None,
) -> dict[str, Any]:
    """OTLP JSON for `POST /api/v1/spans`: one `resourceSpans` entry per
    agent, resource attributes `service.name` + the per-machine §3 identity,
    every other field on the span."""
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        session = sessions.get(span.session_id or "") if span.session_id else None
        raw: dict[str, Any] = {
            "traceId": span.trace_id,
            "spanId": span.span_id,
            "name": span.name,
            "kind": _KIND_MAP.get(span.kind.value, 1),
            "startTimeUnixNano": _ns(span.start_time),
            "endTimeUnixNano": _ns(span.end_time),
            "attributes": _otlp_attrs(span_attributes_for_wire(span, session, capture)),
            "status": {"code": _STATUS_MAP.get(span.status_code.value, 0)},
        }
        if span.parent_span_id:
            raw["parentSpanId"] = span.parent_span_id
        if span.status_message:
            raw["status"]["message"] = span.status_message
        events = _scrub_events(span.events, capture)
        if events:
            raw["events"] = events
        by_agent.setdefault(span.agent_id or "", []).append(raw)

    resource_spans = []
    for agent_id, raws in by_agent.items():
        resource: dict[str, Any] = {"service.name": agent_id or "unknown_service"}
        if install_id:
            resource[TjAttributes.INSTALL_ID] = install_id
        if host_name:
            resource[ResourceAttributes.HOST_NAME] = host_name
        resource_spans.append({
            "resource": {"attributes": _otlp_attrs(resource)},
            "scopeSpans": [{"scope": {"name": "tokenjam.cloud_sync"}, "spans": raws}],
        })
    return {"resourceSpans": resource_spans}


def session_to_wire(session: SessionRecord, *, install_id: str | None) -> dict[str, Any]:
    """One `sessions` row for `POST /api/v1/ledger/sessions` (contracts §6):
    the session's identity, timing and totals, its §4 ledger columns,
    `plan_tier` and the derived `pricing_mode`, and the install it came from."""
    row: dict[str, Any] = {
        "session_id": session.session_id,
        "agent_id": session.agent_id,
        "conversation_id": session.conversation_id,
        "started_at": _iso(session.started_at),
        "ended_at": _iso(session.ended_at),
        "status": session.status,
        "total_cost_usd": session.total_cost_usd,
        "input_tokens": session.input_tokens,
        "output_tokens": session.output_tokens,
        "cache_tokens": session.cache_tokens,
        "cache_write_tokens": session.cache_write_tokens,
        "tool_call_count": session.tool_call_count,
        "error_count": session.error_count,
        "plan_tier": session.plan_tier,
        "pricing_mode": session.pricing_mode,
        "source": session.source,
        "service_namespace": session.service_namespace,
        "service_instance_id": session.service_instance_id,
        "run_id": session.run_id,
        "parent_session_id": session.parent_session_id,
        "bridge_session_id": session.bridge_session_id,
        "install_id": install_id,
    }
    for f in SESSION_CONTEXT_FIELDS:
        row[f] = getattr(session, f)
    return row


def commit_to_wire(commit: SessionCommit) -> dict[str, Any]:
    """One `session_commits` row, the contracts §4 columns verbatim."""
    return {
        "session_id": commit.session_id,
        "commit_sha": commit.commit_sha,
        "repo_remote": commit.repo_remote,
        "confidence": commit.confidence,
        "source": commit.source,
        "author_email": commit.author_email,
        "committed_at": _iso(commit.committed_at),
        "matched_at": _iso(commit.matched_at),
        "match_delta_s": commit.match_delta_s,
    }


# --- HTTP client ------------------------------------------------------------------

class Outcome:
    OK = "ok"
    UNAUTHORIZED = "unauthorized"
    REJECTED = "rejected"        # a 4xx other than 401/429: the receiver refused the body
    UNAVAILABLE = "unavailable"  # 5xx / 429 / network, after the retry schedule


@dataclass(frozen=True)
class PostResult:
    outcome: str
    status: int | None = None
    detail: str = ""
    #: Rows the receiver refused INSIDE a 2xx (the `rejected` count of the
    #: OSS partial-success body); 0 when the body carries none.
    rejected: int = 0


def auth_headers(config: TjConfig) -> dict[str, str]:
    """Exactly the two headers Cloud's ingest auth reads."""
    return {
        ORG_HEADER: config.cloud.org_id.strip(),
        "Authorization": f"Bearer {config.cloud.ingest_key.strip()}",
        "Content-Type": "application/json",
    }


class CloudClient:
    """Posts one body at a time with the retry policy above. `transport` is
    the seam tests use (`httpx.MockTransport`); `sleep` too."""

    def __init__(
        self,
        config: TjConfig,
        *,
        transport: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff: Sequence[float] = RETRY_BACKOFF_S,
    ) -> None:
        import httpx

        self._base = config.cloud.endpoint.rstrip("/")
        self._headers = auth_headers(config)
        self._sleep = sleep
        self._backoff = tuple(backoff)
        self._client = httpx.Client(transport=transport, timeout=REQUEST_TIMEOUT_S)

    def close(self) -> None:
        self._client.close()

    def post(self, path: str, body: Mapping[str, Any]) -> PostResult:
        import httpx

        url = self._base + path
        payload = json.dumps(body, default=str).encode("utf-8")
        attempt = 0
        while True:
            try:
                resp = self._client.post(url, content=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                status, detail = None, f"{type(exc).__name__}: {exc}"
            else:
                status = resp.status_code
                detail = _response_detail(resp)
                if status < 300:
                    return PostResult(Outcome.OK, status, detail, rejected=_rejected_count(resp))
                if status == 401:
                    return PostResult(Outcome.UNAUTHORIZED, status, detail)
                if 400 <= status < 500 and status != 429:
                    return PostResult(Outcome.REJECTED, status, detail)
            if attempt >= len(self._backoff):
                return PostResult(Outcome.UNAVAILABLE, status, detail)
            wait = self._backoff[attempt]
            if status == 429 or status == 503:
                wait = max(wait, _retry_after(resp))
            self._sleep(wait)
            attempt += 1


def _retry_after(resp: Any) -> float:
    try:
        return float(resp.headers.get("Retry-After", "0"))
    except (TypeError, ValueError):
        return 0.0


def _rejected_count(resp: Any) -> int:
    """`rejected` from the OSS partial-success body, bounded by what the
    body actually lists when it lists anything; 0 for any other shape."""
    try:
        data = resp.json()
    except ValueError:
        return 0
    if not isinstance(data, Mapping):
        return 0
    try:
        n = int(data.get("rejected") or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def _response_detail(resp: Any) -> str:
    try:
        data = resp.json()
    except ValueError:
        return (resp.text or "")[:200]
    if isinstance(data, Mapping):
        for key in ("error", "detail", "message"):
            if key in data:
                return str(data[key])[:200]
        rejections = data.get("rejections")
        if isinstance(rejections, list) and rejections:
            reasons = [str(r.get("reason", "")) for r in rejections[:3] if isinstance(r, Mapping)]
            return "; ".join(x for x in reasons if x)[:200]
        return ""
    return str(data)[:200]


# --- The pass ----------------------------------------------------------------------

@dataclass
class SyncReport:
    spans_sent: int = 0
    sessions_sent: int = 0
    commits_sent: int = 0
    rejected: int = 0
    #: Why the pass stopped early, if it did (`unauthorized`, `unavailable`,
    #: or the exception text); None when every stream drained.
    stopped: str | None = None
    skipped_reason: str | None = None

    @property
    def sent_anything(self) -> bool:
        return bool(self.spans_sent or self.sessions_sent or self.commits_sent)


def _columns(cursor: Any) -> list[str]:
    return [d[0] for d in cursor.description]


# Arrival-order columns (migration 25). A row whose stamp is NULL (written by
# a build that predates the column and never touched since the migration
# defaulted it) sorts first and is read once on the first pass.
_SPANS_ALL = "SELECT * FROM spans ORDER BY ingested_at, span_id LIMIT $1"
_SPANS_FROM = ("SELECT * FROM spans WHERE ingested_at >= $1 "
               "ORDER BY ingested_at, span_id LIMIT $2")
_SPANS_AFTER = ("SELECT * FROM spans WHERE ingested_at > $1 "
                "ORDER BY ingested_at, span_id LIMIT $2")
_SESSIONS_ALL = "SELECT * FROM sessions ORDER BY updated_at, session_id LIMIT $1"
_SESSIONS_FROM = ("SELECT * FROM sessions WHERE updated_at >= $1 "
                  "ORDER BY updated_at, session_id LIMIT $2")
_SESSIONS_AFTER = ("SELECT * FROM sessions WHERE updated_at > $1 "
                   "ORDER BY updated_at, session_id LIMIT $2")
_COMMITS_SELECT = ("SELECT session_id, commit_sha, confidence, source, repo_remote, "
                   "author_email, committed_at, matched_at, match_delta_s "
                   "FROM session_commits ")
_COMMITS_ALL = _COMMITS_SELECT + "ORDER BY matched_at, session_id, commit_sha LIMIT $1"
_COMMITS_FROM = (_COMMITS_SELECT + "WHERE matched_at >= $1 "
                 "ORDER BY matched_at, session_id, commit_sha LIMIT $2")
_COMMITS_AFTER = (_COMMITS_SELECT + "WHERE matched_at > $1 "
                  "ORDER BY matched_at, session_id, commit_sha LIMIT $2")


class _Source:
    """Reads the three streams off a direct DuckDB backend, boundary-aware.
    Three literal statements per stream (everything / from the mark inclusive
    / strictly past it); the only parameters are the mark and the page size."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self._sessions: dict[str, SessionRecord | None] = {}

    def session(self, session_id: str | None) -> SessionRecord | None:
        if not session_id:
            return None
        if session_id not in self._sessions:
            self._sessions[session_id] = self.db.get_session(session_id)
        return self._sessions[session_id]

    def _page(self, sql_all: str, sql_from: str, sql_after: str,
              inclusive: bool, hwm: datetime | None, limit: int) -> Any:
        """One cursor over a page: everything (no mark yet), from the mark
        inclusive, or strictly past it."""
        if hwm is None:
            return self.db.conn.execute(sql_all, [limit])
        return self.db.conn.execute(sql_from if inclusive else sql_after, [hwm, limit])

    def spans(self, mark: StreamMark, limit: int) -> list[tuple[NormalizedSpan, datetime]]:
        from tokenjam.core.db import _row_to_span

        def _fetch(inclusive: bool, hwm: datetime | None, n: int) -> list[tuple[NormalizedSpan, datetime]]:
            cur = self._page(_SPANS_ALL, _SPANS_FROM, _SPANS_AFTER, inclusive, hwm, n)
            cols = _columns(cur)
            out = []
            for r in cur.fetchall():
                out.append((_row_to_span(r, cols), dict(zip(cols, r))["ingested_at"]))
            return out

        return _boundary(_fetch, mark, limit, key=lambda t: t[0].span_id, at=lambda t: t[1])

    def sessions(self, mark: StreamMark, limit: int) -> list[tuple[SessionRecord, datetime]]:
        from tokenjam.core.db import _row_to_session

        def _fetch(inclusive: bool, hwm: datetime | None, n: int) -> list[tuple[SessionRecord, datetime]]:
            cur = self._page(_SESSIONS_ALL, _SESSIONS_FROM, _SESSIONS_AFTER,
                             inclusive, hwm, n)
            cols = _columns(cur)
            out = []
            for r in cur.fetchall():
                out.append((_row_to_session(r, cols), dict(zip(cols, r))["updated_at"]))
            return out

        return _boundary(_fetch, mark, limit, key=lambda t: t[0].session_id, at=lambda t: t[1])

    def commits(self, mark: StreamMark, limit: int) -> list[SessionCommit]:
        def _fetch(inclusive: bool, hwm: datetime | None, n: int) -> list[SessionCommit]:
            cur = self._page(_COMMITS_ALL, _COMMITS_FROM, _COMMITS_AFTER, inclusive, hwm, n)
            return [SessionCommit(*r) for r in cur.fetchall()]

        return _boundary(
            _fetch, mark, limit,
            key=lambda c: f"{c.session_id}:{c.commit_sha}", at=lambda c: c.matched_at,
        )


def _boundary(fetch: Callable[[bool, datetime | None, int], list], mark: StreamMark, limit: int, *,
              key: Callable[[Any], str], at: Callable[[Any], datetime | None]) -> list:
    """One page of at most `limit` rows past the mark.

    Reads from the mark INCLUSIVE, over-fetching by the number of ids already
    sent at the mark so the page is still full after those are dropped; if
    the boundary timestamp alone holds more rows than that (an over-fetched
    page that filtering emptied), reads strictly past it instead."""
    hwm = mark.hwm_dt()
    if hwm is None:
        return fetch(True, None, limit)
    sent = set(mark.in_flight_ids)
    rows = fetch(True, hwm, limit + len(sent))
    kept = [r for r in rows if not (at(r) == hwm and key(r) in sent)][:limit]
    if kept or not rows:
        return kept
    return fetch(False, hwm, limit)


def _advance(mark: StreamMark, rows: Sequence[Any], *, key: Callable[[Any], str],
             at: Callable[[Any], datetime | None]) -> None:
    """Move a stream's mark past an acknowledged batch."""
    stamps: list[datetime] = [t for t in (at(r) for r in rows) if t is not None]
    if not stamps:
        return
    newest = max(stamps)
    ids = [key(r) for r in rows if at(r) == newest]
    if mark.hwm_dt() == newest:
        ids = list(dict.fromkeys(mark.in_flight_ids + ids))
    mark.hwm = _iso(newest)
    mark.in_flight_ids = ids


def run_sync(
    db: Any,
    config: TjConfig,
    *,
    client: CloudClient | None = None,
    state_file: Path | None = None,
    install_id: str | None = None,
    host_name: str | None = None,
) -> SyncReport:
    """One forwarding pass over every stream. Never raises on transport or
    receiver trouble; a DuckDB error propagates so the caller's fatal
    classifier sees it (see :func:`start_sync`)."""
    report = SyncReport()
    if not config.cloud.is_active:
        report.skipped_reason = "cloud forwarding is not configured"
        return report
    if not _PASS_LOCK.acquire(blocking=False):
        report.skipped_reason = "a forwarding pass is already running"
        return report
    try:
        return _run_sync_locked(db, config, report, client=client, state_file=state_file,
                                install_id=install_id, host_name=host_name)
    finally:
        _PASS_LOCK.release()


def _run_sync_locked(
    db: Any,
    config: TjConfig,
    report: SyncReport,
    *,
    client: CloudClient | None,
    state_file: Path | None,
    install_id: str | None,
    host_name: str | None,
) -> SyncReport:
    state = load_state(state_file)
    db_path = storage_identity(config)
    if (state.org_id and state.org_id != config.cloud.org_id) or (
        state.db_path and state.db_path != db_path
    ):
        # A different org or store: none of the marks describe what it holds.
        state = SyncState(org_id=config.cloud.org_id, db_path=db_path)
        save_state(state, state_file)
    state.org_id = config.cloud.org_id
    state.db_path = db_path
    if state.disabled_reason:
        report.skipped_reason = state.disabled_reason
        return report
    state.last_attempt_at = utcnow().isoformat()

    own_client = client is None
    client = client or CloudClient(config)
    capture = content_gate(config)
    install = install_id if install_id is not None else ensure_install_id()
    host = host_name if host_name is not None else _host_name()
    source = _Source(db)

    def _send(path: str, body: Mapping[str, Any], mark: StreamMark, rows: Sequence[Any], *,
              key: Callable[[Any], str], at: Callable[[Any], datetime | None],
              rebuild: Callable[[Sequence[Any]], Mapping[str, Any]],
              counter: str) -> tuple[int, bool]:
        """Post `body`; on a refusal split until the bad row is isolated.
        Returns (rows acknowledged, keep going). The mark, the `counter`
        field and the state file all move together, after the ack and
        before anything else, so what is on disk always describes exactly
        the rows the receiver has confirmed."""
        result = client.post(path, body)
        if result.outcome == Outcome.OK:
            # A 2xx is the receiver's ack for the BATCH; the OSS partial
            # success shape inside it names the rows it refused. Those are
            # per-record faults a retry cannot fix (the same body would be
            # refused again), so they are advanced past like any other
            # refused row, and counted, never silently folded into "sent".
            refused = result.rejected
            if refused:
                state.rejected += refused
                report.rejected += refused
                state.last_error = (
                    f"receiver refused {refused} record(s) in an accepted batch"
                    f"{': ' + result.detail if result.detail else ''}"
                )
            else:
                state.last_error = None
            _advance(mark, rows, key=key, at=at)
            sent = len(rows) - refused
            setattr(state, counter, getattr(state, counter) + sent)
            state.last_success_at = utcnow().isoformat()
            save_state(state, state_file)
            return sent, True
        if result.outcome == Outcome.UNAUTHORIZED:
            state.disabled_reason = (
                f"Cloud rejected the ingest key ({result.status}"
                f"{': ' + result.detail if result.detail else ''}). Copy a current key "
                "from Cloud's Connect screen, then: tj init --cloud <key> --org <org>."
            )
            state.disabled_at = utcnow().isoformat()
            state.last_error = state.disabled_reason
            save_state(state, state_file)
            report.stopped = Outcome.UNAUTHORIZED
            return 0, False
        if result.outcome == Outcome.REJECTED:
            if len(rows) > 1:
                half = len(rows) // 2
                sent_a, go = _send(path, rebuild(rows[:half]), mark, rows[:half],
                                   key=key, at=at, rebuild=rebuild, counter=counter)
                if not go:
                    return sent_a, False
                sent_b, go = _send(path, rebuild(rows[half:]), mark, rows[half:],
                                   key=key, at=at, rebuild=rebuild, counter=counter)
                return sent_a + sent_b, go
            # One row the receiver refuses: skip it, count it, move on.
            state.rejected += 1
            report.rejected += 1
            state.last_error = f"receiver refused a record ({result.status}: {result.detail})"
            _advance(mark, rows, key=key, at=at)
            save_state(state, state_file)
            return 0, True
        state.last_error = (
            f"Cloud unavailable ({result.status or 'network'}"
            f"{': ' + result.detail if result.detail else ''}); will retry"
        )
        save_state(state, state_file)
        report.stopped = Outcome.UNAVAILABLE
        return 0, False

    try:
        # Sessions first, so a commit or span batch never names a session
        # the receiver has not seen. Each loop runs until a page comes back
        # EMPTY: a short page is not the end, because the boundary filter can
        # shorten a full page by the rows already sent at the mark.
        while True:
            pairs = source.sessions(state.sessions, BATCH_SIZE)
            if not pairs:
                break

            def _sessions_body(rows: Sequence[tuple[SessionRecord, datetime]]) -> dict[str, Any]:
                return {"sessions": [session_to_wire(s, install_id=install) for s, _ in rows],
                        "session_commits": []}

            n, go = _send(LEDGER_SESSIONS_PATH, _sessions_body(pairs), state.sessions, pairs,
                          key=lambda t: t[0].session_id, at=lambda t: t[1],
                          rebuild=_sessions_body, counter="sessions_sent")
            report.sessions_sent += n
            if not go:
                return report

        while True:
            commits = source.commits(state.session_commits, BATCH_SIZE)
            if not commits:
                break

            def _commits_body(rows: Sequence[SessionCommit]) -> dict[str, Any]:
                return {"sessions": [], "session_commits": [commit_to_wire(c) for c in rows]}

            n, go = _send(LEDGER_SESSIONS_PATH, _commits_body(commits), state.session_commits,
                          commits, key=lambda c: f"{c.session_id}:{c.commit_sha}",
                          at=lambda c: c.matched_at, rebuild=_commits_body,
                          counter="commits_sent")
            report.commits_sent += n
            if not go:
                return report

        while True:
            spans = source.spans(state.spans, BATCH_SIZE)
            if not spans:
                break
            sessions = {sid: source.session(sid)
                        for sid in {s.session_id for s, _ in spans if s.session_id}}

            def _spans_body(rows: Sequence[tuple[NormalizedSpan, datetime]]) -> dict[str, Any]:
                return encode_spans_otlp([s for s, _ in rows], sessions, capture=capture,
                                         install_id=install, host_name=host)

            n, go = _send(SPANS_PATH, _spans_body(spans), state.spans, spans,
                          key=lambda t: t[0].span_id, at=lambda t: t[1],
                          rebuild=_spans_body, counter="spans_sent")
            report.spans_sent += n
            if not go:
                return report
    finally:
        save_state(state, state_file)
        if own_client:
            client.close()
    return report


def _host_name() -> str | None:
    try:
        return socket.gethostname() or None
    except OSError:
        return None


def start_sync(
    db_factory: Callable[[], Any],
    config: TjConfig,
    *,
    on_done: Callable[[SyncReport], None] | None = None,
    state_file: Path | None = None,
) -> threading.Thread | None:
    """Run :func:`run_sync` on a daemon thread with its OWN backend, the way
    the transcript catch-up and the analyzer cycle run. Returns None without
    starting anything when the bridge is not active. Errors are logged, never
    raised; a DuckDB fatal is classified and recovered like every other
    daemon job (Critical Rule 45)."""
    if not config.cloud.is_active:
        return None

    def _run() -> None:
        backend = None
        try:
            backend = db_factory()
            report = run_sync(backend, config, state_file=state_file)
            if report.sent_anything:
                logger.info(
                    "cloud sync forwarded %d span(s), %d session(s), %d commit(s)",
                    report.spans_sent, report.sessions_sent, report.commits_sent,
                )
            if on_done is not None:
                on_done(report)
        except Exception as exc:  # noqa: BLE001 - classified below
            from tokenjam.core.db import handle_if_fatal

            if not handle_if_fatal(exc, what="cloud sync"):
                logger.warning("cloud sync pass failed", exc_info=True)
                _note_error(str(exc), state_file)
        finally:
            from tokenjam.core.db import recover_if_fatal_noted

            recover_if_fatal_noted(what="cloud sync")
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_run, name="tj-cloud-sync", daemon=True)
    thread.start()
    return thread


def _note_error(text: str, state_file: Path | None) -> None:
    state = load_state(state_file)
    state.last_error = text[:300]
    state.last_attempt_at = utcnow().isoformat()
    save_state(state, state_file)


# --- Surfaces ------------------------------------------------------------------------

def _age(iso: str | None) -> str:
    if not iso:
        return "never"
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return "unknown"
    secs = max(0, int((utcnow() - when).total_seconds()))
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def status_summary(config: TjConfig, *, state_file: Path | None = None) -> dict[str, Any]:
    """The bridge as `tj status --json` reports it, and as `status_line`
    renders it. Reads config and the state file only, never the database."""
    cloud = config.cloud
    state = load_state(state_file)
    if not cloud.configured:
        return {"configured": False, "enabled": False, "state": "not_configured"}
    if not cloud.enabled:
        st = "disabled"
    elif state.disabled_reason:
        st = "disabled_by_receiver"
    elif state.last_error and not state.last_success_at:
        st = "unreachable"
    elif state.last_success_at:
        st = "connected"
    else:
        st = "pending"
    return {
        "configured": True,
        "enabled": cloud.enabled,
        "state": st,
        "endpoint": cloud.endpoint,
        "org_id": cloud.org_id,
        "forward_content": cloud.forward_content,
        "spans_sent": state.spans_sent,
        "sessions_sent": state.sessions_sent,
        "commits_sent": state.commits_sent,
        "rejected": state.rejected,
        "last_success_at": state.last_success_at,
        "last_error": state.last_error,
        "disabled_reason": state.disabled_reason,
    }


def status_line(config: TjConfig, *, state_file: Path | None = None) -> str | None:
    """`Cloud: connected · N spans, M sessions sent · last 2m ago`, the
    disable reason, or None when `[cloud]` was never configured (nothing to
    report on a machine that never opted in)."""
    s = status_summary(config, state_file=state_file)
    if not s["configured"]:
        return None
    if s["state"] == "disabled":
        return "Cloud: off (turn forwarding back on with: tj init --cloud <key> --org <org>)"
    if s["state"] == "disabled_by_receiver":
        return f"Cloud: disabled: {s['disabled_reason']}"
    counts = (f"{s['spans_sent']} spans, {s['sessions_sent']} sessions, "
              f"{s['commits_sent']} commits sent")
    if s["rejected"]:
        counts += f", {s['rejected']} refused by the receiver"
    if s["state"] == "unreachable":
        return f"Cloud: unreachable ({s['last_error']}) · {counts}"
    if s["state"] == "pending":
        return f"Cloud: configured, first pass pending · {counts}"
    tail = f" · last {_age(s['last_success_at'])}"
    if s["last_error"]:
        tail += f" · last error: {s['last_error']}"
    return f"Cloud: connected · {counts}{tail}"


def probe(config: TjConfig, *, transport: Any | None = None) -> PostResult:
    """`tj doctor`'s reachability + key check: an EMPTY OTLP body to the spans
    route. Auth runs before the body is read, so a bad key answers 401 and a
    good one 200 with nothing ingested; no telemetry leaves."""
    client = CloudClient(config, transport=transport, backoff=())
    try:
        return client.post(SPANS_PATH, {"resourceSpans": []})
    finally:
        client.close()


__all__ = [
    "BATCH_SIZE",
    "EMISSION_LEAVES",
    "EMISSION_NEVER_BY_DEFAULT",
    "INGEST_KEY_PREFIX",
    "LEDGER_SESSIONS_PATH",
    "ORG_HEADER",
    "SPANS_PATH",
    "SYNC_INTERVAL_MINUTES",
    "CloudClient",
    "Outcome",
    "PostResult",
    "StreamMark",
    "SyncReport",
    "SyncState",
    "auth_headers",
    "commit_to_wire",
    "content_gate",
    "encode_spans_otlp",
    "load_state",
    "parse_ingest_key",
    "probe",
    "reset_state",
    "run_sync",
    "storage_identity",
    "save_state",
    "session_to_wire",
    "span_attributes_for_wire",
    "start_sync",
    "state_path",
    "status_line",
    "status_summary",
]
